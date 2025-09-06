#!/usr/bin/env python3
"""
Compatibility shim for legacy references to 'guardian_agi_min.updated.py'.

The authoritative implementation lives in 'guardian_agi_min.py' (~2k LOC).
This file exists only to avoid breaking scripts or notes that may import or
execute 'guardian_agi_min.updated'. It forwards to guardian_agi_min.main().
"""

from __future__ import annotations
import sys

def main() -> None:
    try:
        import guardian_agi_min as _g
    except Exception as e:  # import failure should be explicit and actionable
        sys.stderr.write(
            f"[guardian.updated] Failed to import guardian_agi_min: {e}\n"
        )
        sys.exit(1)

    entry = getattr(_g, "main", None)
    if callable(entry):
        entry()
    else:
        sys.stderr.write(
            "[guardian.updated] guardian_agi_min.main() not found.\n"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
