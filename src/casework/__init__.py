"""casework -- scoped subject collection and face clustering for investigators.

The tool collects only from sources an analyst has already attributed to an
already-identified subject, and matches only against reference images
enrolled in the case. It builds no index of the general public and has no
"who is this?" operation; see ``scope.py`` and ``faces/match.py``, where
those two limits are implemented rather than merely described.
"""

__version__ = "0.1.0"
