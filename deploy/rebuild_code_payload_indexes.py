#!/usr/bin/env python3
"""
Rebuild nested payload keyword indexes for code collections.

Use this after deploying the nested-key payload-index fix in Helix when
existing code collections still have stale/empty keyword indexes on
metadata.* fields.

Default behavior is dry-run.
"""

from __future__ import annotations

import argparse
import os
from qdrant_client import QdrantClient, models


TARGET_FIELDS = [
    "metadata.language",
    "metadata.path_prefix",
    "metadata.repo_id",
    "metadata.repo_rel_path",
    "metadata.repo",
    "metadata.kind",
    "metadata.symbol",
    "metadata.symbol_path",
    "metadata.imports",
    "metadata.calls",
    "metadata.file_hash",
    "metadata.ingested_at",
    "metadata.last_modified_at",
    "metadata.churn_count",
    "metadata.author_count",
]


def is_code_collection(name: str) -> bool:
    return not (name.endswith("_graph") or name.endswith("_history"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY") or None)
    parser.add_argument("--collection", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=args.url, api_key=args.api_key)
    if args.collection:
        collections = args.collection
    else:
        collections = sorted(c.name for c in client.get_collections().collections if c.name and is_code_collection(c.name))

    print(f"collections={len(collections)} apply={args.apply}")
    for name in collections:
        print(f"[collection] {name}")
        if not args.apply:
            continue
        for field in TARGET_FIELDS:
            try:
                client.delete_payload_index(collection_name=name, field_name=field, wait=True)
            except Exception:
                pass
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print(f"  rebuilt {field}")


if __name__ == "__main__":
    main()
