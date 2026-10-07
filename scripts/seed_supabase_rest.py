import csv
import os
import sys
from pathlib import Path

import requests


TABLES = ("users", "vehicles", "rides", "payments", "ratings")
BATCH_SIZE = 1000


def clean(row: dict[str, str]) -> dict[str, str | None]:
    return {key: (None if value == "" else value) for key, value in row.items()}


def main() -> int:
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    secret_key = os.environ.get("SUPABASE_SECRET_KEY", "")

    if not supabase_url or not secret_key:
        print("Set SUPABASE_URL and SUPABASE_SECRET_KEY before running this script.")
        return 1
    if secret_key.startswith("sb_publishable_") or secret_key.startswith("eyJ"):
        print("SUPABASE_SECRET_KEY must be a Supabase secret key, not a publishable/anon key.")
        print("Find it in Project Settings -> API -> Project API keys -> secret key.")
        return 1

    headers = {
        "apikey": secret_key,
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    data_dir = Path("data")
    for table in TABLES:
        path = data_dir / f"{table}.csv"
        if not path.exists():
            print(f"Missing {path}")
            return 1

        total = 0
        batch: list[dict[str, str | None]] = []
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                batch.append(clean(row))
                if len(batch) >= BATCH_SIZE:
                    total += upload_batch(supabase_url, table, headers, batch)
                    batch = []

        if batch:
            total += upload_batch(supabase_url, table, headers, batch)

        print(f"{table}: {total}")

    return 0


def upload_batch(
    supabase_url: str,
    table: str,
    headers: dict[str, str],
    batch: list[dict[str, str | None]],
) -> int:
    response = requests.post(
        f"{supabase_url}/rest/v1/{table}",
        headers=headers,
        json=batch,
        timeout=60,
    )
    if response.status_code >= 300:
        print(f"{table}: HTTP {response.status_code}")
        print(response.text[:2000])
        raise SystemExit(1)
    return len(batch)


if __name__ == "__main__":
    raise SystemExit(main())
