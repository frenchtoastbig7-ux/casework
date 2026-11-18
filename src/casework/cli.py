"""Command line interface.

The command order mirrors the order the work has to happen in, and the tool
refuses to let you skip a step: you cannot add a source without naming the
subject it belongs to, cannot collect without a live authorisation, and
cannot match without an enrolled reference. Each of those refusals exists
because the corresponding shortcut is what turns scoped collection back into
open-ended searching.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .case import CaseStore, list_cases
from .config import SETTINGS
from .models import Authorisation, Source, Subject, utcnow

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Expected refusals -- an expired authorisation, an out-of-scope URL, a
    # missing model -- are normal operating conditions, not crashes. A stack
    # trace on top of a clear message just buries the message.
    pretty_exceptions_enable=False,
    help="Scoped subject collection and face-clustering workbench for authorised investigators.",
)
case_app = typer.Typer(no_args_is_help=True, help="Create and inspect cases.")
subject_app = typer.Typer(no_args_is_help=True, help="Enrol subjects and reference images.")
source_app = typer.Typer(no_args_is_help=True, help="Declare authorised collection points.")
models_app = typer.Typer(no_args_is_help=True, help="Local face model cache.")
app.add_typer(case_app, name="case")
app.add_typer(subject_app, name="subject")
app.add_typer(source_app, name="source")
app.add_typer(models_app, name="models")


def _slug(value: str) -> str:
    from .config import safe_name

    return safe_name(value)


def _progress(event: str, detail: str) -> None:
    colour = {
        "refused": "yellow", "failed": "red", "done": "bold green",
        "warn": "yellow", "image": "dim", "page": "cyan",
    }.get(event, "white")
    console.print(f"[{colour}]{event:>8}[/] {detail}")


# --------------------------------------------------------------------- case
@case_app.command("new")
def case_new(
    case_id: str = typer.Argument(..., help="Short case handle, e.g. op-fieldmouse"),
    title: str = typer.Option(..., "--title", "-t"),
    holder: str = typer.Option(..., "--holder", help="Who holds the authorisation"),
    instrument: str = typer.Option(..., "--instrument", help="Warrant / engagement / approval ref"),
    expires: str = typer.Option(..., "--expires", help="YYYY-MM-DD; collection stops after this"),
    granted: str = typer.Option("", "--granted", help="YYYY-MM-DD (default: today)"),
    scope_note: str = typer.Option("", "--scope-note"),
) -> None:
    """Create a case. The authorisation is required and it is enforced."""
    def parse(value: str, label: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"--{label} must be a date as YYYY-MM-DD, got {value!r}") from None

    auth = Authorisation(
        holder=holder,
        instrument=instrument,
        granted=parse(granted, "granted") if granted else utcnow().date(),
        expires=parse(expires, "expires"),
        scope_note=scope_note,
    )
    store = CaseStore.create(_slug(case_id), title, auth)
    console.print(
        Panel(
            f"[bold]{title}[/]\n{store.dir}\n\n"
            f"Authorisation {instrument} held by {holder}\n"
            f"Expires {auth.expires} ([bold]{auth.days_left()} days[/])",
            title=f"case {store.id}",
            border_style="red" if auth.days_left() < 0 else "green",
        )
    )
    if auth.days_left() < 0:
        console.print(
            f"[red]This authorisation expired on {auth.expires}.[/] The case can be "
            "reviewed, but `casework crawl` will refuse to collect."
        )
        store.audit.append("case.created_expired", instrument=instrument, expires=auth.expires)


@case_app.command("list")
def case_list() -> None:
    rows = list_cases()
    if not rows:
        console.print("[dim]No cases yet. Create one with `casework case new`.[/]")
        raise typer.Exit()
    table = Table("id", "title", "expires", "subjects", "artifacts")
    for row in rows:
        table.add_row(
            str(row["id"]), str(row["title"]), str(row["expires"]),
            str(row["subjects"]), str(row["artifacts"]),
        )
    console.print(table)


@case_app.command("show")
def case_show(case_id: str) -> None:
    store = CaseStore(_slug(case_id))
    case = store.load()
    stats = store.stats()
    auth = case.authorisation
    left = auth.days_left()
    console.print(
        Panel(
            f"[bold]{case.title}[/]\n"
            f"Authorisation {auth.instrument} ({auth.holder})\n"
            f"Valid {auth.granted} to {auth.expires} "
            f"([{'red' if left < 0 else 'green'}]{left} days[/])\n"
            f"{auth.scope_note}",
            title=f"case {case.id}",
        )
    )
    table = Table("metric", "count")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)

    if case.sources:
        src = Table("source", "subject", "url", "depth")
        for source in case.sources:
            src.add_row(source.id, source.subject_id, source.url, str(source.depth))
        console.print(src)


# ------------------------------------------------------------------ subject
@subject_app.command("add")
def subject_add(
    case_id: str,
    subject_id: str = typer.Option(..., "--id"),
    label: str = typer.Option(..., "--label"),
    note: str = typer.Option("", "--note"),
) -> None:
    store = CaseStore(_slug(case_id))
    case = store.load()
    sid = _slug(subject_id)
    if case.subject(sid):
        console.print(f"[red]Subject {sid!r} already exists.[/]")
        raise typer.Exit(1)
    case.subjects.append(Subject(id=sid, label=label, note=note))
    store.save(case)
    store.audit.append("subject.added", subject=sid, label=label)
    console.print(f"[green]Subject {sid}[/] ({label}) added.")


@subject_app.command("ref")
def subject_ref(
    case_id: str,
    subject_id: str,
    images: list[Path] = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Enrol reference images. Matching is only ever against these."""
    store = CaseStore(_slug(case_id))
    case = store.load()
    subject = case.subject(_slug(subject_id))
    if subject is None:
        console.print(f"[red]No subject {subject_id!r}.[/]")
        raise typer.Exit(1)
    for image in images:
        subject.reference_images.append(store.add_reference(subject.id, image))
    store.save(case)
    store.audit.append("subject.reference_added", subject=subject.id, count=len(images))
    console.print(f"[green]{len(images)}[/] reference image(s) enrolled for {subject.label}.")


# ------------------------------------------------------------------- source
@source_app.command("add")
def source_add(
    case_id: str,
    subject_id: str = typer.Option(..., "--subject", help="Which enrolled subject this belongs to"),
    url: str = typer.Option("", "--url", help="Authorised prefix; the crawl will not leave it"),
    handle: str = typer.Option("", "--handle", help="Handle to expand, e.g. @alice@mastodon.social"),
    platform: str = typer.Option("", "--platform", help="Platform for --handle (see `casework platforms`)"),
    instance: str = typer.Option("", "--instance", help="Instance host for a federated platform"),
    attribution: str = typer.Option(
        ..., "--attribution",
        help="How it was established this source belongs to the subject",
    ),
    depth: int = typer.Option(
        -1, "--depth",
        help="Path segments below the prefix to follow (default: the platform's, else 1)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation of an expanded URL"),
) -> None:
    """Authorise one collection point, by URL or by expanding a handle."""
    store = CaseStore(_slug(case_id))
    case = store.load()
    sid = _slug(subject_id)
    if case.subject(sid) is None:
        console.print(
            f"[red]No subject {sid!r}.[/] Every source must belong to an enrolled subject."
        )
        raise typer.Exit(1)

    if bool(url) == bool(handle):
        console.print("[red]Give exactly one of --url or --handle.[/]")
        raise typer.Exit(1)

    resolved_depth = depth
    if handle:
        if not platform:
            console.print("[red]--handle needs --platform.[/] See `casework platforms`.")
            raise typer.Exit(1)
        from .platforms import expand

        url, plat = expand(platform, handle, instance or None)
        if resolved_depth < 0:
            resolved_depth = plat.default_depth

        # An expanded URL is shown before it is stored. The expansion is a
        # guess at what the analyst meant, and a wrong guess authorises
        # collection against the wrong account.
        rating = {"open": "green", "partial": "yellow", "blocked": "red"}[plat.collection]
        console.print(f"  {plat.name} handle expands to [bold]{url}[/]")
        console.print(f"  collection prospects: [{rating}]{plat.collection}[/] -- {plat.note}")
        if not yes and not typer.confirm("  Authorise this URL?", default=True):
            raise typer.Exit(1)

    if resolved_depth < 0:
        resolved_depth = 1

    source = Source(
        id=f"src-{len(case.sources) + 1:03d}", subject_id=sid, url=url,
        attribution=attribution, depth=resolved_depth,
    )
    from .scope import Prefix

    prefix = Prefix.from_url(source.url, source.id, sid, resolved_depth)
    case.sources.append(source)
    store.save(case)
    store.audit.append(
        "source.added", source=source.id, subject=sid, url=url,
        attribution=attribution, depth=resolved_depth,
        **({"handle": handle, "platform": platform} if handle else {}),
    )
    shown = prefix.host + ("/" + "/".join(prefix.segments) if prefix.segments else "")
    console.print(
        f"[green]{source.id}[/] authorises [bold]{shown}[/] "
        f"and up to {resolved_depth} level(s) beneath it. Nothing else on that host."
    )


@source_app.command("list")
def source_list(case_id: str) -> None:
    case = CaseStore(_slug(case_id)).load()
    table = Table("id", "subject", "url", "depth", "on", "attribution")
    for source in case.sources:
        table.add_row(
            source.id, source.subject_id, source.url, str(source.depth),
            "yes" if source.enabled else "no", source.attribution[:60],
        )
    console.print(table)


# -------------------------------------------------------------------- scope
@app.command("scope")
def scope_check(case_id: str, url: str) -> None:
    """Ask whether a URL would be collected, without fetching anything."""
    from .crawl.runner import preview_scope

    verdict = preview_scope(_slug(case_id), url)
    if verdict["allowed"]:
        console.print(
            f"[green]in scope[/] via {verdict['source_id']} (subject {verdict['subject_id']}), "
            f"depth {verdict['depth_used']}/{verdict['depth_limit']}"
        )
    else:
        console.print(f"[yellow]out of scope[/] {verdict['reason']}")


# -------------------------------------------------------------------- crawl
@app.command("crawl")
def crawl_cmd(
    case_id: str,
    source: list[str] = typer.Option(None, "--source", help="Limit to these source ids"),
    max_pages: int = typer.Option(200, "--max-pages"),
    max_images: int = typer.Option(2000, "--max-images"),
    ignore_robots: bool = typer.Option(
        False, "--ignore-robots",
        help="Override robots.txt. Logged as its own audit entry.",
    ),
    refresh: bool = typer.Option(
        True, "--refresh/--no-refresh",
        help="Re-read pages already collected, to pick up newly posted images",
    ),
) -> None:
    """Collect from the case's authorised sources."""
    from .crawl.runner import crawl

    run = crawl(
        _slug(case_id), source_ids=list(source) if source else None,
        respect_robots=not ignore_robots, max_pages=max_pages,
        max_images=max_images, refresh_pages=refresh, progress=_progress,
    )
    console.print(
        Panel(
            f"collected {run.fetched}\nalready held {run.skipped}\n"
            f"refused (out of scope) {run.refused}\nfailed {run.failed}",
            title=f"run {run.id}", border_style="green",
        )
    )
    for note in run.notes:
        console.print(f"[yellow]note[/] {note}")


# -------------------------------------------------------------------- faces
@app.command("faces")
def faces_cmd(
    case_id: str,
    reset: bool = typer.Option(False, "--reset", help="Rebuild all embeddings from scratch"),
) -> None:
    """Detect and embed faces in everything collected so far."""
    from .faces.pipeline import process

    result = process(_slug(case_id), reset=reset, progress=_progress)
    console.print(
        f"[green]{result['faces']}[/] faces in {result['images']} images "
        f"({result['unreadable']} unreadable)"
    )


@app.command("cluster")
def cluster_cmd(
    case_id: str,
    threshold: float = typer.Option(None, "--threshold", help="Cosine distance; lower is stricter"),
) -> None:
    """Group faces by apparent identity. Clusters are not identities."""
    from .faces.cluster import cluster_case

    result = cluster_case(_slug(case_id), threshold)
    console.print(
        f"[green]{result['clusters']}[/] clusters over {result['faces']} faces "
        f"({result.get('singletons', 0)} singletons, largest {result.get('largest', 0)})"
    )


@app.command("match")
def match_cmd(
    case_id: str,
    threshold: float = typer.Option(None, "--threshold", help="Cosine similarity; higher is stricter"),
) -> None:
    """Attribute clusters to enrolled subjects."""
    from .faces.match import match_case

    result = match_case(_slug(case_id), threshold, progress=_progress)
    table = Table("cluster", "verdict", "subject", "faces", "score", "margin")
    for decision in result["decisions"]:
        table.add_row(
            str(decision["cluster"]),
            str(decision["verdict"]),
            str(decision.get("subject") or decision.get("candidates") or "-"),
            str(decision["size"]),
            str(decision.get("score", "-")),
            str(decision.get("margin", "-")),
        )
    console.print(table)
    console.print(
        f"threshold {result['threshold']}  "
        f"[green]{result['matched']} matched[/]  "
        f"{result['unknown']} unknown  "
        f"[yellow]{result['contested']} contested[/]"
    )
    console.print(
        "[dim]Scores are cosine similarity, not probabilities. "
        "Run `casework calibrate` on known material before relying on them.[/]"
    )


@app.command("calibrate")
def calibrate_cmd(
    case_id: str,
    subject_id: str,
    positives: Path = typer.Option(..., "--pos", exists=True, help="Dir of images WITH the subject"),
    negatives: Path = typer.Option(..., "--neg", exists=True, help="Dir of images WITHOUT them"),
) -> None:
    """Measure score separation on material you have ground truth for."""
    from .faces.match import calibrate

    def images(folder: Path) -> list[Path]:
        return [
            p for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        ]

    result = calibrate(_slug(case_id), _slug(subject_id), images(positives), images(negatives))
    table = Table("metric", "value")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


# ----------------------------------------------------------------- analysis
@app.command("graph")
def graph_cmd(case_id: str, min_weight: int = typer.Option(1, "--min-weight")) -> None:
    """Co-occurrence: who appears in the same photographs as whom."""
    from .analysis.cooccur import cooccurrence, unknowns_near_subjects

    graph = cooccurrence(_slug(case_id), min_weight)
    table = Table("cluster A", "cluster B", "shared photos")
    for edge in graph["edges"][:40]:
        table.add_row(str(edge["a"]), str(edge["b"]), str(edge["photos"]))
    console.print(table)

    leads = unknowns_near_subjects(_slug(case_id))
    if leads:
        console.print("\n[bold]Unnamed clusters appearing with enrolled subjects[/]")
        lead_table = Table("cluster", "photos", "shared with")
        for lead in leads[:20]:
            lead_table.add_row(
                str(lead["cluster"]), str(lead["photos"]),
                ", ".join(f"{k} x{v}" for k, v in lead["with_subjects"].items()),
            )
        console.print(lead_table)


@app.command("report")
def report_cmd(
    case_id: str,
    redact: bool = typer.Option(
        True, "--redact/--no-redact",
        help="Blur faces not attributed to an enrolled subject",
    ),
) -> None:
    """Write a self-contained HTML case report."""
    from .report import write_report

    path = write_report(_slug(case_id), redact_unmatched=redact)
    console.print(f"[green]Report written[/] {path}")


@app.command("audit")
def audit_cmd(
    case_id: str,
    verify: bool = typer.Option(False, "--verify", help="Recompute the hash chain"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    store = CaseStore(_slug(case_id))
    if verify:
        ok, detail = store.audit.verify()
        console.print(f"[{'green' if ok else 'red'}]{detail}[/]")
        raise typer.Exit(0 if ok else 1)
    table = Table("time", "event", "actor", "detail")
    for entry in store.audit.entries(limit):
        detail = ", ".join(f"{k}={v}" for k, v in list(entry["data"].items())[:3])
        table.add_row(entry["ts"][:19], entry["event"], entry.get("actor", ""), detail[:80])
    console.print(table)


# ------------------------------------------------------------------- models
@models_app.command("status")
def models_status() -> None:
    from .modelmgr import status

    table = Table("model", "cached", "size", "purpose")
    for row in status():
        table.add_row(
            str(row["name"]), "yes" if row["cached"] else "no",
            f"{int(row['size']) / 1e6:.1f} MB" if row["size"] else "-", str(row["purpose"]),
        )
    console.print(table)
    console.print(f"[dim]{SETTINGS.model_cache}[/]")


@models_app.command("fetch")
def models_fetch() -> None:
    from .modelmgr import ensure_models

    last = {"name": "", "pct": -1}

    def show(name: str, done: int, total: int) -> None:
        pct = int(done * 100 / max(total, 1))
        if pct != last["pct"] or name != last["name"]:
            last.update(name=name, pct=pct)
            console.print(f"[dim]{name}[/] {pct}%", end="\r")

    paths = ensure_models(show)
    console.print(f"\n[green]Models ready[/] ({len(paths)} files in {SETTINGS.model_cache})")


# ---------------------------------------------------------------------- gui
@app.command("gui")
def gui_cmd(
    case_id: str = typer.Argument("", help="Case to open (optional)"),
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Serve the review interface on localhost."""
    import uvicorn

    from .server import create_app

    console.print(f"[green]casework[/] http://{host}:{port}/{('#' + case_id) if case_id else ''}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command("platforms")
def platforms_cmd(
    detail: str = typer.Option("", "--detail", help="Show one platform's full note"),
) -> None:
    """Platforms whose handles `source add --handle` can expand.

    Expansion is string templating only: it never checks whether a handle
    exists, and it cannot search a handle across platforms.
    """
    from .platforms import catalogue, get

    if detail:
        plat = get(detail)
        console.print(Panel(
            f"[bold]{plat.name}[/]\n\n"
            f"template   {plat.template}\n"
            f"example    {plat.example}\n"
            f"depth      {plat.default_depth}\n"
            f"collection {plat.collection}\n\n{plat.note}",
            title=plat.key,
        ))
        if plat.key == "mastodon":
            console.print(
                "\n[dim]A bare profile source (https://instance/@alice) is collected "
                "through the instance REST API:\n"
                "  /api/v1/accounts/lookup            resolve the account\n"
                "  /api/v1/accounts/<id>/statuses     only_media, paginated\n\n"
                "That gives full-resolution attachments, the real posting time, and "
                "boosts excluded -- none of which the rendered page offers. Media is "
                "accepted only from hosts this instance uses for the account's own "
                "avatar and header, which bounds it without hardcoding a CDN.\n\n"
                "If the instance refuses the API (some require authentication for "
                "public timelines), collection falls back to crawling the profile "
                "page and says so in the run notes and the audit log. The fallback "
                "cannot tell an authored post from a boost and records only when the "
                "material was collected, not when it was posted.[/]"
            )
        if plat.key == "bluesky":
            console.print(
                "\n[dim]bsky.app renders client-side, so a Bluesky source is collected "
                "through the public XRPC API instead of by crawling the profile page. "
                "Three hosts are involved and each is pinned:\n"
                "  bsky.app             what you declare as the source\n"
                "  public.api.bsky.app  the endpoint actually queried\n"
                "  cdn.bsky.app         where the images live\n\n"
                "Every CDN URL carries the DID of the account that uploaded the image, "
                "so an image belonging to anyone but this source's account is refused "
                "before it is fetched -- reposts and quoted posts included. Every API "
                "response is archived alongside the images it produced.[/]"
            )
        return

    table = Table("key", "platform", "example", "collects", "note")
    colour = {"open": "green", "partial": "yellow", "blocked": "red"}
    for row in catalogue():
        rating = str(row["collection"])
        table.add_row(
            str(row["key"]), str(row["name"]), str(row["example"]),
            f"[{colour[rating]}]{rating}[/]", str(row["note"])[:64],
        )
    console.print(table)
    console.print(
        "[dim]'collects' is what a plain unauthenticated fetch gets back, not whether "
        "the account exists. A 'blocked' source is still authorised and still crawled "
        "-- it will simply return little or nothing.[/]"
    )


@app.command("version")
def version_cmd() -> None:
    console.print("casework 0.1.0")
    console.print(f"[dim]home {SETTINGS.root}[/]")
    console.print(f"[dim]network {'enabled' if SETTINGS.allow_network else 'DISABLED'}[/]")


EXPECTED = (
    PermissionError,   # AuthorisationError, ScopeError
    FileNotFoundError,
    FileExistsError,
    ValueError,
    KeyError,
)


def main() -> None:
    from pydantic import ValidationError

    from .config import OfflineError
    from .faces.recognise import ModelsUnavailable

    try:
        app()
    except ValidationError as exc:
        # Pydantic's default rendering buries the useful sentence under a URL
        # and a type code; the validators here are written to be read.
        for error in exc.errors():
            field = ".".join(str(p) for p in error["loc"]) or "input"
            console.print(f"[red]{field}[/] {error['msg'].removeprefix('Value error, ')}")
        raise SystemExit(1) from None
    except (*EXPECTED, OfflineError, ModelsUnavailable) as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/] Work already saved is kept; re-run to resume.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
