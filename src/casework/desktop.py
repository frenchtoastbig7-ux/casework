"""Desktop shell: the local server in a native window.

Same server, same loopback binding -- pywebview just removes the step where
an analyst has a case-review interface sitting in a browser tab alongside
everything else they have open.
"""
from __future__ import annotations

import socket
import threading
import time

import uvicorn

# Absolute, not relative: PyInstaller runs this file as the top-level
# ``__main__`` script with no parent package, so ``from .server import …``
# raises ImportError inside the frozen bundle even though it works under
# ``python -m casework.desktop``. Importing by full package name works in
# both, since the package is on the path either way.
from casework.server import create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    try:
        import webview
    except ImportError:
        raise SystemExit(
            "pywebview is not installed. Install the desktop extra "
            "(pip install 'casework[app]') or run `casework gui` and use a browser."
        ) from None

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    # Wait for the socket rather than sleeping a fixed interval: on a cold
    # start the import of OpenCV and scikit-learn can take a few seconds, and
    # a window that loads before the server is up shows a connection error.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.15)

    webview.create_window("casework", f"http://127.0.0.1:{port}/", width=1280, height=860)
    webview.start()
    server.should_exit = True


if __name__ == "__main__":
    main()
