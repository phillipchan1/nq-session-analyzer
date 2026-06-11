import json
from pathlib import Path

APPROVED_PATH = Path(__file__).resolve().parent / "approved_patterns.json"


def append_approved_pattern(pattern: dict):
    """Append a single approved pattern record to approved_patterns.json."""
    if APPROVED_PATH.exists():
        try:
            data = json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
    else:
        data = []

    if not isinstance(data, list):
        data = []

    data.append(pattern)
    APPROVED_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    # This is a stub used by the automation job; it expects the caller to import.
    print(f"approved file: {APPROVED_PATH}")
