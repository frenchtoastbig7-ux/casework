"""Append-only, hash-chained audit log.

Every collection and every identity assertion writes a line here. The chain
(each entry carries the hash of the previous one) means a deleted or edited
line is detectable, which is the difference between a log and a record: a
plain log proves nothing about what was *removed* from it.

This is not tamper-proofing -- anyone who can rewrite the file can recompute
the chain. It is tamper-*evidence* against the realistic case, which is a
line quietly dropped or back-dated later, not a forensic adversary.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open("rb") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    last = json.loads(raw).get("hash", last)
                except json.JSONDecodeError:
                    continue
        return last

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "actor": os.environ.get("CASEWORK_ACTOR") or os.environ.get("USER", "unknown"),
            "data": data,
            "prev": self._last_hash(),
        }
        entry["hash"] = hashlib.sha256(_canonical(entry)).hexdigest()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:] if limit else rows

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, human-readable detail)."""
        prev = GENESIS
        for i, entry in enumerate(self.entries(), start=1):
            claimed = entry.get("hash")
            body = {k: v for k, v in entry.items() if k != "hash"}
            if body.get("prev") != prev:
                return False, f"Entry {i} does not chain to its predecessor (log was edited)."
            if hashlib.sha256(_canonical(body)).hexdigest() != claimed:
                return False, f"Entry {i} hash does not match its contents."
            prev = claimed
        return True, f"Chain intact across {len(self.entries())} entries."
