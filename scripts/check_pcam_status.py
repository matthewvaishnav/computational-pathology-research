"""Check current PCam training status from status heartbeat file."""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PCam training status")
    parser.add_argument(
        "--status-file",
        type=str,
        default="logs/pcam/training_status.json",
        help="Path to training status JSON file",
    )
    args = parser.parse_args()

    raw_status_file = Path(args.status_file)
    status_file = raw_status_file if raw_status_file.is_absolute() else REPO_ROOT / raw_status_file
    if not status_file.exists():
        print(f"Status file not found: {status_file}")
        return

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            status = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid status file JSON: {exc}")
        return
    except Exception as exc:
        print(f"Failed to read status file: {exc}")
        return

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
