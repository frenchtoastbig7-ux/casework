# PyInstaller spec — builds "Casework.app" for macOS.
#
#   pyinstaller packaging/casework-mac.spec --noconfirm
#
# The bundle carries the interpreter and every dependency, so it runs on a
# Mac with no Python, Homebrew or pip. The two face models (YuNet ~0.2 MB,
# SFace ~39 MB) are deliberately NOT bundled: the app downloads them on first
# use into ~/.casework/models, and collection/analysis of already-held
# material works without them.
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (injected)
VERSION = "0.1.0"

hiddenimports: list[str] = []

# Server internals resolved dynamically at runtime, which static analysis misses.
for pkg in ("uvicorn", "anyio", "pydantic", "pydantic_core", "websockets",
            "httptools", "uvloop", "httpx", "httpcore", "h11", "starlette"):
    hiddenimports += collect_submodules(pkg)

# Starlette looks up python-multipart under both spellings when parsing forms.
hiddenimports += ["multipart", "python_multipart"]

# scikit-learn's clustering pulls compiled scipy/joblib helpers by attribute
# access that PyInstaller cannot see. Collect the trees rather than guess the
# handful of leaf modules -- getting one wrong disables clustering silently.
for pkg in ("sklearn", "scipy", "joblib", "threadpoolctl", "numpy"):
    hiddenimports += collect_submodules(pkg)

# BeautifulSoup's CSS-select support lives in a separate package.
hiddenimports += ["bs4", "soupsieve"]

# pywebview's Cocoa backend imports pyobjc even though casework's own code
# never does; without these the window cannot be created inside the bundle.
for pkg in ("objc", "Foundation", "AppKit", "WebKit", "CoreFoundation",
            "Quartz", "PyObjCTools"):
    hiddenimports += collect_submodules(pkg)
hiddenimports += collect_submodules("webview")

# casework imports its own heavy modules lazily, inside request handlers and
# the crawl runner, so a build without a given feature still starts. That same
# laziness hides them from static analysis -- name them or the frozen app
# loses face detection, or Bluesky/Mastodon collection, with no error until
# the button is pressed.
hiddenimports += collect_submodules("casework")
hiddenimports += [
    "casework.server", "casework.desktop",
    "casework.crawl.runner", "casework.crawl.fetch", "casework.crawl.extract",
    "casework.crawl.collector", "casework.crawl.bluesky", "casework.crawl.mastodon",
    "casework.faces.pipeline", "casework.faces.recognise",
    "casework.faces.cluster", "casework.faces.match",
    "casework.analysis.cooccur", "casework.report", "casework.modelmgr",
    "casework.platforms", "casework.scope",
    "cv2",
]

datas = [
    # The review GUI. Without this the window loads a blank page.
    (os.path.join(REPO, "src", "casework", "web"), "casework/web"),
]
datas += collect_data_files("webview")
datas += collect_data_files("sklearn")
datas += collect_data_files("cv2")

# Only whole packages that nothing in the dependency graph imports. Kept
# short on purpose: an over-eager exclude of a scipy/sklearn submodule looks
# safe and silently breaks clustering, for a saving of tens of megabytes.
excludes = [
    "tkinter", "matplotlib", "IPython", "notebook", "jupyter",
    "pytest", "_pytest", "torch", "torchvision", "onnxruntime",
]

a = Analysis(
    [os.path.join(REPO, "src", "casework", "desktop.py")],
    pathex=[os.path.join(REPO, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Casework",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                       # windowed app, no terminal
    icon=os.path.join(REPO, "packaging", "casework.icns"),
    disable_windowed_traceback=False,
    target_arch=None,                    # build for the host architecture
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="Casework",
)
app = BUNDLE(
    coll,
    name="Casework.app",
    icon=os.path.join(REPO, "packaging", "casework.icns"),
    bundle_identifier="local.casework.workbench",
    version=VERSION,
    info_plist={
        "CFBundleName": "Casework",
        "CFBundleDisplayName": "Casework",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "LSUIElement": False,
        "NSRequiresAquaSystemAppearance": False,
        "NSHumanReadableCopyright": "For authorised investigative use. Runs locally.",
    },
)
