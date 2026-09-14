"""Collection pass over a case's authorised sources.

The loop is a breadth-first walk that never leaves ``Scope``. Two properties
are worth stating because they are what separate this from a web crawler:

* the frontier is seeded **only** from declared sources, never from
  discovered content, and every candidate is re-checked against scope before
  it is fetched, so a page cannot widen its own authorisation;
* an off-scope link is *counted and logged as refused* rather than silently
  dropped, because "the crawler ignored 4,000 links to other people's
  profiles" is exactly the fact an audit needs to be able to establish.

Runs are resumable. Already-collected URLs are skipped on a re-run, so an
interrupted collection is restarted with the same command.
"""
from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from datetime import date

from ..case import CaseStore
from ..config import ScopeError
from ..models import Artifact, CrawlRun, utcnow
from ..scope import Scope, is_asset_url, normalise
from .extract import image_urls, page_links, page_title
from .fetch import Fetcher, FetchResult

Progress = Callable[[str, str], None]


def _noop(_event: str, _detail: str) -> None:
    return None


def _store_artifact(
    store: CaseStore, result: FetchResult, source_id: str, subject_id: str,
    referer: str | None, posted: str | None = None,
) -> tuple[Artifact, bool]:
    """Write bytes to content-addressed storage and build the artifact row.

    Returns ``(artifact, is_new_blob)``. The same image reached by two URLs
    hashes identically, so the bytes are written once while both URLs get
    their own provenance row -- which is itself a finding worth keeping,
    since it establishes the same photograph appeared in two places.
    """
    suffix = result.suffix
    path = store.blob_path(result.sha256, suffix)
    is_new = not path.exists()
    if is_new:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.content)

    artifact = Artifact(
        id=uuid.uuid4().hex[:16],
        source_id=source_id,
        subject_id=subject_id,
        url=result.url,
        referer=referer,
        fetched=utcnow(),
        status=result.status,
        content_type=result.content_type,
        sha256=result.sha256,
        bytes=len(result.content),
        path=str(path.relative_to(store.dir)),
        is_image=result.is_image,
        posted=posted,
    )
    return artifact, is_new


def _api_sources(active) -> list[tuple[object, object]]:
    """Pair each source with a factory for the collector that should run it.

    A source that matches no API collector is left to the HTML crawler, which
    is the correct default -- most of the web is not an API.
    """
    from .bluesky import BlueskyCollector, actor_from_source_url
    from .mastodon import MastodonCollector, account_from_source_url

    pairs: list[tuple[object, object]] = []
    for source in active:
        actor = actor_from_source_url(source.url)
        if actor is not None:
            pairs.append((source, lambda f, a=actor: BlueskyCollector(f, a)))
            continue
        account = account_from_source_url(source.url)
        if account is not None:
            instance, user = account
            pairs.append(
                (source, lambda f, i=instance, u=user: MastodonCollector(f, i, u))
            )
    return pairs


def _collect_api(
    store: CaseStore, case, run: CrawlRun, source, fetcher: Fetcher,
    collector, max_images: int, progress: Progress,
) -> tuple[list[Artifact], bool]:
    """Drive one API collector for one source.

    Returns ``(artifacts, resolved)``. ``resolved`` is False when the account
    could not be looked up at all, which the caller may treat as a reason to
    fall back to the HTML crawler on platforms where that works.

    Scope is enforced by the collector's own ``check_media_url``, which
    differs per platform -- a DID comparison on Bluesky, a derived host set on
    Mastodon -- so this function stays ignorant of how authorship is proven
    and only insists that it is.
    """
    from .collector import CollectorError

    name = collector.platform
    collected: list[Artifact] = []
    api_artifacts: list[Artifact] = []

    def archive(result: FetchResult) -> None:
        artifact, _new = _store_artifact(store, result, source.id, source.subject_id, None)
        api_artifacts.append(artifact)
        store.audit.append(
            f"{name}.api", run=run.id, url=result.url, status=result.status,
            sha256=result.sha256, subject=source.subject_id,
        )

    collector.on_response = archive
    try:
        profile = collector.resolve()
    except CollectorError as exc:
        run.failed += 1
        run.notes.append(f"{source.id}: {exc}")
        store.audit.append(f"{name}.failed", run=run.id, source=source.id, error=str(exc))
        progress("failed", f"{source.id}: {exc}")
        return api_artifacts, False

    label = getattr(collector, "handle", None) or getattr(collector, "acct", source.url)
    progress("page", f"{name} {label}")
    store.audit.append(
        f"{name}.resolved", run=run.id, source=source.id, actor=collector.actor,
        identity=getattr(collector, "did", None) or getattr(collector, "account_id", None),
    )

    seen = {normalise(a.url) for a in case.artifacts}
    count = 0

    def take(item) -> None:
        nonlocal count
        if count >= max_images:
            return
        candidate = normalise(item.url)
        if candidate in seen:
            run.skipped += 1
            return
        try:
            collector.check_media_url(item.url)
        except ScopeError as exc:
            run.refused += 1
            store.audit.append(f"{name}.refused", run=run.id, url=item.url, reason=str(exc))
            progress("refused", str(exc))
            return

        progress("image", item.url)
        result = fetcher.get(item.url, referer=item.post_url)
        if result is None or not result.ok or not result.is_image:
            run.failed += 1
            return
        seen.add(candidate)
        count += 1
        artifact, _new = _store_artifact(
            store, result, source.id, source.subject_id, item.post_url,
            posted=item.created_at,
        )
        collected.append(artifact)
        run.fetched += 1
        store.audit.append(
            "fetch.ok", run=run.id, url=result.url, status=result.status,
            sha256=result.sha256, bytes=len(result.content),
            subject=source.subject_id, kind="image", referer=item.post_url,
            post_uri=item.post_uri, posted=item.created_at,
            sensitive=item.sensitive, via=f"{name}-api",
        )

    try:
        for item in collector.profile_media(profile):
            take(item)
        for item in collector.post_media():
            take(item)
    except CollectorError as exc:
        run.notes.append(f"{source.id}: walk stopped -- {exc}")
        store.audit.append(f"{name}.failed", run=run.id, source=source.id, error=str(exc))
        progress("failed", f"{source.id}: {exc}")

    if collector.parse_looks_wrong():
        warning = (
            f"{source.id}: the {name} API returned {collector.entries_seen} media post(s) "
            "but no image URL could be read from any of them. The response shape has "
            "probably changed; treat this source as NOT collected rather than as empty."
        )
        run.notes.append(warning)
        store.audit.append(
            f"{name}.parse_warning", run=run.id, source=source.id,
            entries=collector.entries_seen,
        )
        progress("failed", warning)
    if count >= max_images:
        run.notes.append(f"{source.id}: image budget of {max_images} reached.")
    progress("done", f"{name} {label}: {count} image(s)")
    return api_artifacts + collected, True


def crawl(
    case_id: str,
    source_ids: list[str] | None = None,
    respect_robots: bool = True,
    max_pages: int = 200,
    max_images: int = 2000,
    refresh_pages: bool = True,
    progress: Progress = _noop,
    today: date | None = None,
) -> CrawlRun:
    store = CaseStore(case_id)
    case = store.load()

    # Hard gate. An expired mandate stops collection before a single request
    # is made, and the refusal is recorded.
    try:
        case.authorisation.check(today)
    except Exception as exc:
        store.audit.append("crawl.refused", reason=str(exc))
        raise

    active = [s for s in case.sources if s.enabled and (not source_ids or s.id in source_ids)]
    if not active:
        raise ValueError(
            "No enabled sources to collect from. Add one with `casework source add` -- "
            "every source must name the subject it belongs to and how that was established."
        )

    scope = Scope(
        [
            p
            for p in Scope.from_case(case).prefixes
            if not source_ids or p.source_id in source_ids
        ]
    )

    run = CrawlRun(id=uuid.uuid4().hex[:12], started=utcnow(), sources=[s.id for s in active])
    case.runs.append(run)
    store.save(case)
    store.audit.append(
        "crawl.started",
        run=run.id,
        sources=[{"id": s.id, "url": s.url, "subject": s.subject_id} for s in active],
        respect_robots=respect_robots,
    )
    if not respect_robots:
        # Deliberately its own entry: an override of a site's stated wishes
        # should be visible in the record on its own line, not buried as a
        # field inside the run that used it.
        store.audit.append("crawl.robots_override", run=run.id,
                           note="robots.txt was not honoured for this run")

    # Assets already held are never refetched: an image at a stable URL does
    # not change. Pages are refetched by default, because the whole point of
    # a second run is to pick up what the subject has posted since the first,
    # and a page skipped is a page whose new images are never discovered.
    seen_assets = {normalise(a.url) for a in case.artifacts if a.is_image}
    seen_pages = {normalise(a.url) for a in case.artifacts if not a.is_image}
    held_pairs = {(normalise(a.url), a.sha256) for a in case.artifacts}
    frontier: deque[tuple[str, int]] = deque()
    queued: set[str] = set()

    # Some sources are collected through an API rather than by crawling HTML.
    # Bluesky must be -- bsky.app is a client-side shell with no images in it.
    # Mastodon merely does better that way, so its API failing is a reason to
    # fall back to the page rather than to give up.
    api_sources = _api_sources(active)
    api_ids = {s.id for s, _ in api_sources}

    for source in active:
        if source.id in api_ids:
            continue
        seed = normalise(source.url)
        frontier.append((seed, 0))
        queued.add(seed)

    pages_done = 0
    images_done = 0
    new_artifacts: list[Artifact] = []

    with Fetcher(respect_robots=respect_robots) as fetcher:
        for source, make_collector in api_sources:
            collector = make_collector(fetcher)
            produced, resolved = _collect_api(
                store, case, run, source, fetcher, collector,
                max_images - images_done, progress,
            )
            new_artifacts.extend(produced)
            images_done += sum(1 for a in produced if a.is_image)
            case.artifacts.extend(new_artifacts)
            new_artifacts = []
            store.save(case)

            if not resolved and collector.platform == "mastodon":
                # The instance would not answer the API -- it may require
                # authentication for public timelines. Mastodon's HTML is
                # real, so fall back rather than abandon the source, and say
                # so: a silent downgrade would hide that the collection is
                # now missing post timestamps and boost detection.
                seed = normalise(source.url)
                frontier.append((seed, 0))
                queued.add(seed)
                note = f"{source.id}: Mastodon API unavailable, falling back to page crawl."
                run.notes.append(note)
                store.audit.append("mastodon.fallback", run=run.id, source=source.id)
                progress("refused", note)

        while frontier and pages_done < max_pages:
            url, depth = frontier.popleft()

            try:
                prefix = scope.check_page(url)
            except ScopeError as exc:
                run.refused += 1
                progress("refused", str(exc))
                continue

            if url in seen_pages and not refresh_pages:
                run.skipped += 1
                continue

            if fetcher.blocked_by_robots(url):
                run.refused += 1
                run.notes.append(f"robots.txt disallows {url}")
                store.audit.append("fetch.robots_blocked", run=run.id, url=url)
                progress("refused", f"robots.txt disallows {url}")
                continue

            progress("page", url)
            result = fetcher.get(url)
            pages_done += 1

            if result is None or not result.ok:
                run.failed += 1
                status = result.status if result else 0
                store.audit.append("fetch.failed", run=run.id, url=url, status=status)
                progress("failed", f"{status} {url}")
                continue

            final_url = normalise(result.url)
            unchanged = (final_url, result.sha256) in held_pairs
            if unchanged:
                # Byte-identical to what is already held. Record the
                # re-observation in the log but do not add a second artifact
                # row, or a weekly re-collection would grow the case file
                # without adding anything to it.
                run.skipped += 1
                store.audit.append(
                    "fetch.unchanged", run=run.id, url=result.url, sha256=result.sha256,
                )
            else:
                artifact, _new = _store_artifact(
                    store, result, prefix.source_id, prefix.subject_id, None
                )
                new_artifacts.append(artifact)
                held_pairs.add((final_url, result.sha256))
                run.fetched += 1
                store.audit.append(
                    "fetch.ok", run=run.id, url=result.url, status=result.status,
                    sha256=result.sha256, bytes=len(result.content),
                    subject=prefix.subject_id, kind="image" if result.is_image else "page",
                )

            seen_pages.add(final_url)
            if result.is_image:
                seen_assets.add(final_url)
                images_done += 1
                continue

            if not result.content_type.split(";")[0].strip().lower().endswith(
                ("html", "xhtml+xml")
            ):
                continue

            html = result.content.decode(errors="replace")
            title = page_title(html)
            if title:
                progress("title", title)

            # Images first: they are the point of the collection, and a page
            # budget exhausted before its pictures were pulled is a wasted run.
            for img_url in image_urls(html, result.url):
                if images_done >= max_images:
                    break
                candidate = normalise(img_url)
                if candidate in seen_assets or not is_asset_url(candidate):
                    continue
                try:
                    scope.check_asset(candidate, result.url)
                except ScopeError:
                    run.refused += 1
                    continue
                if fetcher.blocked_by_robots(candidate):
                    run.refused += 1
                    continue

                progress("image", candidate)
                img = fetcher.get(candidate, referer=result.url)
                if img is None or not img.ok or not img.is_image:
                    run.failed += 1
                    continue

                seen_assets.add(normalise(img.url))
                images_done += 1
                img_artifact, _is_new = _store_artifact(
                    store, img, prefix.source_id, prefix.subject_id, result.url
                )
                new_artifacts.append(img_artifact)
                run.fetched += 1
                store.audit.append(
                    "fetch.ok", run=run.id, url=img.url, status=img.status,
                    sha256=img.sha256, bytes=len(img.content),
                    subject=prefix.subject_id, kind="image", referer=result.url,
                )

            allowed_links, refused_links = scope.partition_links(
                result.url, page_links(html, result.url)
            )
            if refused_links:
                run.refused += len(refused_links)
                store.audit.append(
                    "links.refused", run=run.id, page=result.url,
                    count=len(refused_links), sample=refused_links[:5],
                )
                progress(
                    "refused",
                    f"{len(refused_links)} link(s) off this page were outside scope",
                )
            if depth < prefix.depth:
                for link in allowed_links:
                    if link not in queued:
                        queued.add(link)
                        frontier.append((link, depth + 1))

            # Persist as we go. A long collection killed at hour two should
            # keep everything it had already fetched.
            case.artifacts.extend(new_artifacts)
            new_artifacts = []
            store.save(case)

    case.artifacts.extend(new_artifacts)
    if frontier:
        run.notes.append(
            f"Page budget of {max_pages} reached with {len(frontier)} URLs still queued."
        )
    run.finished = utcnow()
    store.save(case)
    store.audit.append(
        "crawl.finished", run=run.id, fetched=run.fetched, skipped=run.skipped,
        failed=run.failed, refused=run.refused,
    )
    progress(
        "done",
        f"{run.fetched} collected, {run.skipped} already held, "
        f"{run.refused} refused as out of scope, {run.failed} failed",
    )
    return run


def preview_scope(case_id: str, url: str) -> dict[str, object]:
    """Answer 'would this URL be collected?' without fetching anything.

    Exists because the honest answer to "what will this crawl touch?" should
    be available before the crawl, not reconstructed from its logs after.
    """
    store = CaseStore(case_id)
    case = store.load()
    scope = Scope.from_case(case)
    try:
        prefix = scope.check_page(url)
    except ScopeError as exc:
        return {"allowed": False, "reason": str(exc)}
    return {
        "allowed": True,
        "source_id": prefix.source_id,
        "subject_id": prefix.subject_id,
        "depth_used": prefix.depth_of(url),
        "depth_limit": prefix.depth,
    }
