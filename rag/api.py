from __future__ import annotations

import json
import sys

from rag.service import run_rag_query


def main() -> None:
    raw = sys.stdin.read().strip()
    payload = json.loads(raw or '{}')
    query = str(payload.get('query') or '').strip()
    top_k = int(payload.get('top_k') or 6)
    graph_expand_k = int(payload.get('graph_expand_k') or 3)
    compress = bool(payload.get('compress', True))
    law_ids = payload.get('law_ids') or []
    if isinstance(law_ids, str):
        law_ids = [item.strip() for item in law_ids.split(',') if item.strip()]
    result = run_rag_query(
        query,
        top_k=max(1, min(top_k, 15)),
        graph_expand_k=max(0, min(graph_expand_k, 6)),
        law_ids={str(item).strip() for item in law_ids if str(item).strip()} or None,
        compress=compress,
    )
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
