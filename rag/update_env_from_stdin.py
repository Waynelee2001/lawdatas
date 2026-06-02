from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    updates = json.load(sys.stdin)
    env_path = Path(".env")
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen: set[str] = set()
    output: list[str] = []

    for line in existing:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0]
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
