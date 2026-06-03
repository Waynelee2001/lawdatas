from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _write_job(job_path: Path, payload: dict) -> None:
    payload["updated_at"] = time.time()
    tmp_path = job_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, job_path)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        print("usage: python -m rag.agent_job_worker JOB_ID JOB_FILE QUESTION_FILE", file=sys.stderr)
        return 2

    job_id, job_file, question_file = argv
    job_path = Path(job_file)
    question_path = Path(question_file)
    now = time.time()
    payload = {
        "job_id": job_id,
        "status": "running",
        "pid": os.getpid(),
        "created_at": now,
        "started_at": now,
        "updated_at": now,
    }

    try:
        with open(question_path, "r", encoding="utf-8") as f:
            question_payload = json.load(f)
        question = str(question_payload.get("question") or "").strip()
        verbose = bool(question_payload.get("verbose", False))
        payload["question"] = question
        _write_job(job_path, payload)

        from rag.agent import run_agent_query

        result = run_agent_query(question, verbose=verbose)
        payload.update(
            {
                "status": "done",
                "result": {
                    "answer": result.get("answer", ""),
                    "rounds": result.get("rounds", 0),
                    "tool_calls": result.get("tool_calls", []),
                },
            }
        )
        _write_job(job_path, payload)
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error": f"Agent 查询失败: {str(exc)[:800]}",
            }
        )
        _write_job(job_path, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
