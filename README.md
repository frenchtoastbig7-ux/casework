# casework

A scoped collection and face-clustering workbench for investigators working
under a defined mandate. You give it people you have **already identified**
and sources you have **already attributed to them**; it collects from exactly
those sources, groups the faces it finds, and tells you who appears with
whom.

It runs entirely on your own hardware. There is no hosted version and no
account.

## Scope

**This is not a face search engine.** There is no index of the general
public, no crawler that wanders, and no "who is this?" operation — not as a
disabled feature, but as an absent one. Ask it about a face it has no
enrolled reference for and the answer is *unknown*, permanently.

What it does instead:

- collects from URL prefixes you declare, each tied to a named subject and to
  a written account of how you established that link
- refuses, counts and logs every candidate URL outside those prefixes
- groups the faces in what it collected, so you learn how many distinct
  people are in the material and which of them recur
- compares those groups **only** against reference images you enrolled
- surfaces co-occurrence — who appears in photographs with whom — which is
  usually the actual finding
- records every action in a hash-chained log, and every collection against an
  authorisation with an expiry date it enforces

It is built for people working under some authority that permits the work.
Collecting imagery of identified people engages legal and ethical obligations
that vary by jurisdiction and by the mandate you hold. Those are yours to
satisfy. There is deliberately no hosted deployment.

---

## Why it is built this way

A face-matching tool fails in a specific and dangerous way: it produces a
confident, precise, wrong identification, and the number next to it does the
analyst's thinking for them. Four design choices push against that.

**Scope is a data structure, not a policy.** A source authorises its own URL
prefix and nothing else on the host, matched segment-wise so `/users/alice`
cannot authorise `/users/alice-smith`. Discovered links never widen
authorisation. `scope.py` is small and heavily tested because everything else
rests on it.

**Authorisation has an expiry and the expiry is enforced.** Authority to
collect is nearly always time-boxed and the box is nearly always missed,
because nothing in the toolchain normally knows about it. Here collection
stops on the date, and the refusal is logged.

**Identity decisions happen per cluster, and must beat the runner-up.** One
face at an awkward angle is a weak observation; forty across a collection is
a strong one. And scoring above threshold for two enrolled subjects is not a
match — it is a warning that your reference images cannot separate them. That
returns `contested` rather than silently picking the higher number.

**Unknown is a normal result.** Most clusters in a real collection are
bystanders. The tool says so rather than reaching for the nearest enrolled
name.

Scores are cosine similarities. They are not probabilities and they do not
mean "96% certain". `casework calibrate` exists so you can measure the score
separation on your own material — compressed reposts, CCTV stills, group
shots at distance — instead of trusting a number measured on a benchmark that
looks nothing like your case.

---

## Install

### macOS app (recommended)

Casework ships as a self-contained `.dmg` — a native Dock app, no terminal
and no Python required. Open the disk image, drag **Casework** onto
Applications, and launch it. Because it is not notarised by an identified
developer, the *first* launch needs a right-click → **Open** rather than a
double-click; after that it opens normally.

The app is the same tool as the CLI with the review GUI in a native window
instead of a browser tab, and both share `~/.casework`, so a case made in one
is visible in the other.

### From source

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e "."
.venv/bin/casework models fetch     # ~39 MB, once; everything then runs offline
```

The two ONNX models (YuNet for detection, SFace for embeddings) come from the
OpenCV Zoo and are cached under `~/.casework/models` — outside any case, since
they are shared and are not case material. `CASEWORK_MODELS` moves that
cache; `CASEWORK_HOME` moves the cases.

### Building the .dmg

```bash
uv pip install --python .venv -e ".[app]"
bash packaging/build_app.sh      # -> dist/Casework.app  (PyInstaller, ~256 MB)
bash packaging/make_dmg.sh       # -> dist/Casework.dmg  (~112 MB, compressed)
```

The bundle carries the interpreter and every dependency; the face models are
still downloaded on first use, not bundled. The build is ad-hoc signed, not
notarised — hence the right-click → Open on first launch. The icon is
generated from `packaging/make_icon.py` rather than committed as a binary.

### Notarising (Developer ID)

To ship a `.dmg` that opens with a plain double-click on any Mac, sign it with
a Developer ID and notarise it. Two one-time setup steps handle secrets, so
they are yours to run, not the build's:

1. **Install your certificate.** In an Apple Developer account, create a
   *Developer ID Application* certificate and install it — with its private
   key — into the login keychain. Confirm it is there:

   ```bash
   security find-identity -v -p codesigning   # must list "Developer ID Application: …"
   ```

2. **Store your notary credentials once.** Use an app-specific password
   (appleid.apple.com → Sign-In and Security) or an App Store Connect API
   key. This is the step that touches your Apple password, so run it yourself:

   ```bash
   xcrun notarytool store-credentials casework-notary \
     --apple-id "you@example.com" --team-id "TEAMID" \
     --password "app-specific-password"
   ```

Then build and notarise in one command:

```bash
bash packaging/build_app.sh       # produce the app (ad-hoc)
bash packaging/notarize.sh        # re-sign with Developer ID, notarise, staple
```

`notarize.sh` signs every nested library inside-out with the hardened runtime
and `packaging/entitlements.plist`, submits the app to Apple and waits for the
verdict, staples the ticket, builds the `.dmg` from the stapled app, notarises
and staples the `.dmg` too, and finally verifies both with `spctl` and
`stapler validate`. The entitlements lift only library-validation — a frozen
Python bundle ships unsigned third-party `.dylib`/`.so` that the hardened
runtime would otherwise refuse at launch — and grant no network, disk or
device access.

## A case, end to end

```bash
casework case new op-fieldmouse \
  --title "Operation Fieldmouse" \
  --holder "DI Okafor" --instrument "WARR-2026-114" --expires 2026-12-01

casework subject add op-fieldmouse --id alice --label "Alice Doe"
casework subject ref op-fieldmouse alice ~/exhibits/alice-custody.jpg

casework source add op-fieldmouse --subject alice \
  --url https://example.org/users/alice \
  --attribution "handle matches seized message header; profile photo matches exhibit 4"

# or expand a handle instead of typing the URL
casework source add op-fieldmouse --subject alice \
  --platform mastodon --handle "@alice@infosec.exchange" \
  --attribution "handle matches seized message header; profile photo matches exhibit 4"

casework scope op-fieldmouse https://example.org/users/bob   # -> out of scope

casework crawl op-fieldmouse
casework faces op-fieldmouse
casework cluster op-fieldmouse
casework match op-fieldmouse
casework graph op-fieldmouse
casework report op-fieldmouse
```

Then review in the GUI:

```bash
casework gui op-fieldmouse
```

## Handles

`casework platforms` lists the 20 platforms whose handles can be expanded.
`--handle` accepts what people actually paste — `@alice`, `alice`, or a
pasted profile URL — and shows you the resulting URL for confirmation before
anything is authorised.

```bash
casework source add op-fieldmouse --subject alice \
  --platform bluesky --handle alice.bsky.social --attribution "…"
```

Two things this deliberately is not:

**It is not a lookup.** Expansion is string templating. It never asks a
platform whether a handle exists, so a wrong handle produces a source that
404s on first collection rather than an error at add time.

**It is not enumeration.** There is no way to take one username and find
which platforms carry it. Sweeping a handle across platforms is a discovery
operation, and discovery is what turns scoped collection back into
open-ended searching. `tests/test_platforms.py` asserts no such function
exists.

Federated platforms (Mastodon, Lemmy, Pixelfed) require the instance,
embedded as `@alice@instance` or passed as `--instance`. This is not
pedantry: the same local name on another instance is a different person, so
there is no safe default to guess.

### What actually collects

Producing a correct profile URL is easy; getting images back from it is not.
Every platform carries a rating, shown in `casework platforms` and in the
GUI as you type:

| | Platforms |
|---|---|
| **open** — collects well | Mastodon, Bluesky, Lemmy, Pixelfed, Flickr, GitHub |
| **partial** — gated, truncated or JS-dependent | Reddit, Tumblr, VK, Telegram, Twitch, YouTube, Pinterest |
| **blocked** — expect nothing | X, Instagram, Facebook, Threads, TikTok, LinkedIn, Snapchat |

A `blocked` source is still authorised and still crawled — it will simply
return little or nothing. The rating is about the fetch, not about whether
the account exists.

## Mastodon

A bare profile source (`https://instance/@alice`) is collected through the
instance REST API rather than by scraping the page. Unlike Bluesky this is an
upgrade rather than a rescue — Mastodon's HTML is real — but the API is
meaningfully better:

- `only_media=true` asks for exactly the posts worth collecting
- attachments carry the full-resolution URL, not the size the page rendered
- `created_at` is when the subject **posted**, which is what a timeline needs;
  it is stored on the artifact, and cluster locations sort by it. The HTML
  path leaves it empty rather than substituting the collection time — a
  timeline built from `fetched` would say every photograph in a ten-year
  account appeared on the afternoon of the crawl
- `reblog` distinguishes authored from boosted, so other people's images are
  refusable rather than merely improbable
- pagination reaches the whole history, not the first page

**The authorship check is weaker than Bluesky's, and that is worth knowing.**
Mastodon has no DID-in-URL, because an instance may serve media from whatever
object store its admin configured. So the bound is derived from the
instance's own behaviour: the hosts serving *this account's* avatar and
header are the hosts trusted to serve its posts, plus the instance itself.
With `status.account.id` matching and `reblog` absent, that gives authorship
at the post level and a bounded host set at the media level — "this account
posted it, and it came from where this instance serves its media", not "this
account uploaded these exact bytes".

If an instance refuses the API — some require authentication for public
timelines — collection **falls back to crawling the profile page** and says
so in the run notes and as a `mastodon.fallback` audit entry. The fallback
cannot tell an authored post from a boost and records only collection time,
so a source that fell back is a weaker collection, not an equivalent one.

A source narrowed to a single status (`/@alice/12345`) or to a remote handle
keeps using the HTML crawler, so deliberately narrow scope stays narrow
rather than being widened to the whole account by the API path.

## Bluesky

`bsky.app` renders client-side, so crawling a profile page returns an empty
shell. A Bluesky source is therefore collected through the public XRPC API
instead — declare it exactly like any other source and the crawler dispatches
on its own:

```bash
casework source add op-fieldmouse --subject alice \
  --platform bluesky --handle alice.bsky.social --attribution "…"
casework crawl op-fieldmouse
```

Three hosts are involved and each is pinned: `bsky.app` is what you declare,
`public.api.bsky.app` is queried, `cdn.bsky.app` serves the images.

That fan-out is why this is a collector rather than a scope exception — and
it ends up **stronger** than the HTML crawler's rule. Every Bluesky CDN URL
contains the DID of the account that uploaded the image:

```
https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc…/bafkrei…@jpeg
```

so each image is checked against the DID resolved from the declared source,
and refused *before it is fetched* if it does not match. Where the HTML
crawler asks "did an in-scope page link to this?" — a heuristic — this asks
"did this subject upload it?" and gets a structural answer. A quote-post
carrying someone else's photograph cannot enter this subject's case file even
though it sits in their feed.

Reposts are skipped on the same reasoning: an author feed contains material
the subject amplified but did not create. For a quote-post, only the media
the subject attached is read, never the quoted post's own images.

Every XRPC response is archived as an artifact alongside the images it
produced, because an image's provenance should not stop at "the API said so"
with nothing recording what the API actually said.

Handles are reassignable domains, so pin the `did:plc:…` where identity
matters — `--handle did:plc:…` expands fine and the collection is pinned to
the DID regardless.

One caveat worth stating plainly: the JSON shape is coded against the AT
Protocol lexicon and tested against fixtures, not against the live service.
If Bluesky changes an embed shape, this collector returns *zero* images
rather than wrong ones — a safe failure, but an invisible one. So the crawl
also counts: if the feed returns media posts and none yields a readable image
URL, the run reports it and writes a `bluesky.parse_warning` to the audit
log. Treat that source as **not collected**, not as empty.

## The GUI

`casework gui` serves a review interface on loopback (`casework-desktop` puts
the same thing in a native window). It shows the collection as a grid of
people rather than a ranked list of hits, because "there are 14 distinct
people here and this one recurs in 41 photographs" is the shape of the
question. Each cluster card carries a cohesion bar — a low bar on a large
cluster is the signal that two people have been merged, and the reason to
tighten the threshold.

Naming a cluster is a human act. The decision is recorded per face (so it
survives re-clustering at a different threshold), it overrides the matcher,
and it lands in the audit log with the reason you typed.

## Re-collection

Re-running `crawl` refreshes pages already collected and skips assets already
held, so a weekly pass picks up newly posted images without refetching the
archive. Byte-identical pages are logged as re-observed rather than stored
again. Everything is content-addressed, so the same photograph found at two
URLs is stored once and keeps both provenance records — which is itself a
finding.

## Reports

`casework report` writes a self-contained HTML file. Faces not attributed to
an enrolled subject are blurred by default: a collection about two people
routinely contains bystanders who are nobody's business, and a report that
leaves them visible becomes an unscoped disclosure the moment it is
forwarded. `--no-redact` exists and is recorded in the log.

Every image carries its source URL, referring page, collection time and
content hash inline.

## Working offline

Set `CASEWORK_OFFLINE=1` and nothing can reach the network. Collection
refuses to start; detection, clustering, matching and reporting all run
normally, because they are local computation over material you already hold.

## Audit

```bash
casework audit op-fieldmouse --verify
```

Each log entry carries the hash of the previous one, so an edited or deleted
line is detectable. This is tamper-evidence, not tamper-proofing: anyone who
can rewrite the file can recompute the chain. It defends against the
realistic case — a line quietly dropped later — not a forensic adversary.

## Limits

- Clusters are hypotheses about sameness, not identities.
- A match means an embedding is close to an enrolled reference. It is not a
  confidence level and it is not corroboration.
- Faces below ~48px are dropped, because a tiny face produces an embedding
  that is confidently wrong rather than usefully vague.
- The recogniser has demographic error-rate differences, as all of them do.
  Calibrate on material resembling your case, and treat any single
  attribution as a lead requiring corroboration.
- Nothing here substitutes for evidence.

## Tests

```bash
.venv/bin/python -m pytest
```

`tests/test_scope.py` is the one to read first: it is the specification of
what this tool will and will not fetch. `tests/test_faces.py` pins the three
match verdicts — matched, unknown, contested — against constructed
embeddings; its ONNX integration tests skip if the models are not cached.
