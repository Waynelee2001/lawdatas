import datetime
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

# Serverless defaults to /tmp; persistent hosts can mount these paths to a disk.
DB_FILE = os.environ.get("USERS_DB_FILE", "/tmp/users.json")
CODES_FILE = os.environ.get("CODES_DB_FILE", "/tmp/codes.json")
AGENT_JOB_DIR = Path(os.environ.get("AGENT_JOB_DIR", "/tmp/law_agent_jobs"))
AGENT_JOB_TTL_SECONDS = int(os.environ.get("AGENT_JOB_TTL_SECONDS", "7200"))

# ============================================================
# 核心密钥 —— 请勿泄露！
# ============================================================
SECRET_KEY = os.environ.get("LAW_ACTIVATION_SECRET_KEY", "LawFaKao@2024_HMAC_K3y_#Z9!mX")

# 字符集：32个无歧义字符（去掉 0/O/1/I 防止混淆）
CHARSET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


# ============================================================
# 激活码核心算法
# ============================================================


def _to_base32_str(n: int, length: int) -> str:
    """将整数 n 编码为 CHARSET 表示的 length 位字符串"""
    result = []
    for _ in range(length):
        result.append(CHARSET[n % 32])
        n //= 32
    return "".join(reversed(result))


def generate_key(prefix: str = "") -> str:
    """
    生成10位激活码，格式：XXXXX-XXXXX
    - 随机载荷：7位（CHARSET随机字符）
    - 校验码：  3位（HMAC-SHA256签名的Base-32编码）
    - 最终展示：中间加"-"分隔，便于阅读和输入
    """
    payload = "".join(secrets.choice(CHARSET) for _ in range(7))
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], "big") % (32**3)
    checksum = _to_base32_str(n, 3)
    raw = payload + checksum  # 共10位
    return f"{raw[:5]}-{raw[5:]}"  # 展示为 XXXXX-XXXXX


def validate_key(key_input: str):
    """
    校验激活码格式与签名。
    返回 (True, 标准化后的10位码) 或 (False, 错误说明)
    """
    # 标准化：去掉"-"、空格，转大写
    key = key_input.strip().upper().replace("-", "").replace(" ", "")

    if len(key) != 10:
        return False, f"激活码长度错误（应为10位，当前 {len(key)} 位）"

    for c in key:
        if c not in CHARSET:
            return False, f"含有不合法字符：{c}"

    payload = key[:7]
    checksum = key[7:]

    # 重新计算期望签名
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], "big") % (32**3)
    expected = _to_base32_str(n, 3)

    if checksum != expected:
        return False, "激活码无效，请确认后重新输入"

    return True, key


# ============================================================
# 数据库 I/O
# ============================================================


def _init_file(path, default):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f)


def load_users():
    _init_file(DB_FILE, {})
    with open(DB_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_users(users):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)


def load_codes():
    _init_file(CODES_FILE, {})
    with open(CODES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_codes(codes):
    with open(CODES_FILE, "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=4, ensure_ascii=False)


# ============================================================
# API 路由
# ============================================================


@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"code": 400, "msg": "邮箱和密码不能为空"}), 400

    users = load_users()
    if username in users:
        return jsonify({"code": 400, "msg": "该邮箱已被注册"}), 400

    users[username] = {
        "password_hash": generate_password_hash(password),
        "activated": False,
        "activated_at": None,
        "code_used": None,
    }
    save_users(users)
    return jsonify({"code": 200, "msg": "注册成功"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"code": 400, "msg": "邮箱和密码不能为空"}), 400

    users = load_users()
    user = users.get(username)
    if user and check_password_hash(user["password_hash"], password):
        token = str(uuid.uuid4())
        return jsonify(
            {
                "code": 200,
                "msg": "登录成功",
                "token": token,
                "activated": user.get("activated", False),
            }
        )
    return jsonify({"code": 401, "msg": "邮箱账号或密码错误"}), 401


@app.route("/redeem", methods=["POST"])
def redeem():
    data = request.json or {}
    username = data.get("username", "").strip()
    code_input = data.get("code", "").strip()

    if not username or not code_input:
        return jsonify({"code": 400, "msg": "参数不完整"}), 400

    # 1. 校验格式 + HMAC 签名
    ok, result = validate_key(code_input)
    if not ok:
        return jsonify({"code": 400, "msg": result}), 400
    code = result  # 标准化后的10位码（无"-"）

    # 2. 用户是否存在
    users = load_users()
    if username not in users:
        return jsonify({"code": 404, "msg": "用户不存在"}), 404

    # 3. 该账号是否已激活
    if users[username].get("activated"):
        return jsonify({"code": 400, "msg": "您的账号已激活，无需重复兑换"}), 400

    # 4. 激活码是否被使用过（一码一用）
    codes = load_codes()
    if code in codes:
        return jsonify({"code": 400, "msg": "该激活码已被使用，请换一个"}), 400

    # 5. 写入记录，激活账号
    codes[code] = {"used_by": username, "used_at": datetime.datetime.now().isoformat()}
    save_codes(codes)

    users[username]["activated"] = True
    users[username]["activated_at"] = datetime.datetime.now().isoformat()
    users[username]["code_used"] = code
    save_users(users)

    return jsonify({"code": 200, "msg": "激活成功！即刻享受完整功能"})


@app.route("/change_password", methods=["POST"])
def change_password():
    data = request.json or {}
    username = data.get("username", "").strip()
    old_pwd = data.get("old_password", "").strip()
    new_pwd = data.get("new_password", "").strip()

    if not username or not old_pwd or not new_pwd:
        return jsonify({"code": 400, "msg": "参数不完整"}), 400

    users = load_users()
    user = users.get(username)
    if not user or not check_password_hash(user["password_hash"], old_pwd):
        return jsonify({"code": 401, "msg": "原密码错误"}), 401

    users[username]["password_hash"] = generate_password_hash(new_pwd)
    save_users(users)
    return jsonify({"code": 200, "msg": "密码修改成功"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"code": 200, "msg": "ok"})


@app.route("/api/rag/health", methods=["GET"])
def rag_health():
    try:
        from rag.config import settings

        qdrant_ready = None
        qdrant_error = ""
        if settings.qdrant_url:
            try:
                from rag.index import build_qdrant_client

                build_qdrant_client().get_collection(settings.qdrant_collection)
                qdrant_ready = True
            except Exception as exc:
                qdrant_ready = False
                qdrant_error = str(exc)[:240]

        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "rag_importable": True,
                    "vector_store": "remote" if settings.qdrant_url else "local",
                    "qdrant_collection": settings.qdrant_collection,
                    "qdrant_ready": qdrant_ready,
                    "qdrant_error": qdrant_error,
                    "has_siliconflow_key": bool(settings.siliconflow_api_key),
                    "has_chat_key": bool(settings.chat_api_key),
                },
            }
        )
    except Exception as exc:
        return jsonify({"code": 500, "msg": f"RAG 配置检查失败: {exc}"}), 500


def _env_fingerprint(name: str) -> dict:
    value = os.getenv(name, "")
    stripped = value.strip()
    return {
        "configured": bool(stripped),
        "length": len(stripped),
        "raw_length": len(value),
        "sha256_12": hashlib.sha256(stripped.encode()).hexdigest()[:12]
        if stripped
        else "",
        "starts_with_sk": stripped.startswith("sk-"),
        "contains_equals": "=" in stripped,
        "has_outer_whitespace": value != stripped,
    }


@app.route("/api/rag/env-fingerprint", methods=["GET"])
def rag_env_fingerprint():
    return jsonify(
        {
            "code": 200,
            "msg": "ok",
            "data": {
                "siliconflow_api_key": _env_fingerprint("SILICONFLOW_API_KEY"),
                "deepseek_api_key": _env_fingerprint("DEEPSEEK_API_KEY"),
                "siliconflow_embedding_model": os.getenv(
                    "SILICONFLOW_EMBEDDING_MODEL", ""
                ),
                "deepseek_chat_model": os.getenv("DEEPSEEK_CHAT_MODEL", ""),
                "deepseek_query_analysis_model": os.getenv(
                    "DEEPSEEK_QUERY_ANALYSIS_MODEL", ""
                ),
            },
        }
    )


def _run_agent_query_sync(question: str, verbose: bool = False) -> dict:
    try:
        from rag.agent import run_agent_query

        return run_agent_query(question, verbose=verbose)
    except ModuleNotFoundError:
        venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        if not venv_python.exists():
            raise
        proc = subprocess.run(
            [str(venv_python), "-m", "rag.agent", question],
            text=True,
            capture_output=True,
            check=True,
            timeout=300,
        )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"answer": proc.stdout.strip(), "rounds": 1, "tool_calls": []}


def _run_agent_job_inline(
    job_id: str, job_path: Path, question: str, verbose: bool = False
) -> None:
    payload = {
        "job_id": job_id,
        "status": "running",
        "inline": True,
        "worker_pid": os.getpid(),
        "question": question,
        "created_at": time.time(),
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _write_agent_job(job_id, payload)
    try:
        result = _run_agent_query_sync(question, verbose=verbose)
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
    except Exception as exc:
        logger.exception("inline agent job failed")
        payload.update(
            {
                "status": "error",
                "error": f"Agent 查询失败: {str(exc)[:800]}",
            }
        )
    _write_agent_job(job_id, payload)


def _trim_agent_jobs(now: float | None = None) -> None:
    now = now or time.time()
    cutoff = now - AGENT_JOB_TTL_SECONDS
    AGENT_JOB_DIR.mkdir(parents=True, exist_ok=True)
    for path in AGENT_JOB_DIR.glob("*.json"):
        if path.name.endswith(".question.json"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                question_path = path.with_suffix(".question.json")
                question_path.unlink(missing_ok=True)
                log_path = path.with_suffix(".log")
                log_path.unlink(missing_ok=True)
        except OSError:
            continue


def _safe_agent_job_id(job_id: str) -> str:
    safe = "".join(ch for ch in str(job_id or "") if ch.isalnum() or ch in "-_")
    return safe if safe == str(job_id or "") and safe else ""


def _agent_job_path(job_id: str) -> Path | None:
    safe = _safe_agent_job_id(job_id)
    if not safe:
        return None
    return AGENT_JOB_DIR / f"{safe}.json"


def _write_agent_job(job_id: str, payload: dict) -> None:
    AGENT_JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = _agent_job_path(job_id)
    if path is None:
        raise ValueError("invalid job id")
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _read_agent_job(job_id: str) -> dict | None:
    path = _agent_job_path(job_id)
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _agent_job_snapshot(job_id: str) -> dict | None:
    payload = _read_agent_job(job_id)
    if not payload:
        return None
    payload["job_id"] = job_id
    if payload.get("status") in {"queued", "running"}:
        if payload.get("inline"):
            worker_pid = payload.get("worker_pid")
            if worker_pid and int(worker_pid) != os.getpid():
                payload["status"] = "error"
                payload["updated_at"] = time.time()
                payload["error"] = "Agent 后台任务所在服务已重启，请重新发起查询。"
                _write_agent_job(job_id, payload)
        elif not _pid_alive(payload.get("pid")):
            payload["status"] = "error"
            payload["updated_at"] = time.time()
            payload["error"] = "Agent 后台任务进程已退出，请重新发起查询。"
            _write_agent_job(job_id, payload)
    started_at = payload.get("started_at") or payload.get("created_at")
    payload["elapsed_seconds"] = round(max(0.0, time.time() - float(started_at)), 1)
    return payload


def _start_agent_job(question: str, verbose: bool = False) -> str:
    _trim_agent_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    AGENT_JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_path = _agent_job_path(job_id)
    if job_path is None:
        raise ValueError("invalid job id")
    question_path = job_path.with_suffix(".question.json")
    log_path = job_path.with_suffix(".log")
    with open(question_path, "w", encoding="utf-8") as f:
        json.dump({"question": question, "verbose": verbose}, f, ensure_ascii=False)
    payload = {
        "status": "queued",
        "question": question,
        "log_path": str(log_path),
        "created_at": now,
        "updated_at": now,
    }
    _write_agent_job(job_id, payload)
    inline_default = os.getenv("AGENT_BOUNDED", "0")
    if os.getenv("AGENT_JOB_INLINE", inline_default) == "1":
        payload.update(
            {
                "status": "running",
                "inline": True,
                "worker_pid": os.getpid(),
                "started_at": time.time(),
                "updated_at": time.time(),
            }
        )
        _write_agent_job(job_id, payload)
        thread = threading.Thread(
            target=_run_agent_job_inline,
            args=(job_id, job_path, question, verbose),
            daemon=True,
            name=f"agent-job-{job_id[:8]}",
        )
        thread.start()
        return job_id
    log_file = None
    try:
        log_file = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rag.agent_job_worker",
                job_id,
                str(job_path),
                str(question_path),
            ],
            cwd=str(Path(__file__).resolve().parent),
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        payload.update(
            {
                "status": "running",
                "pid": proc.pid,
                "started_at": time.time(),
                "updated_at": time.time(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "updated_at": time.time(),
                "error": f"Agent 后台任务启动失败: {str(exc)[:400]}",
            }
        )
    finally:
        if log_file:
            log_file.close()
    _write_agent_job(job_id, payload)
    return job_id


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k") or 6)
    graph_expand_k = int(data.get("graph_expand_k") or 3)
    compress = bool(data.get("compress", True))
    law_ids = data.get("law_ids") or []

    if not query:
        return jsonify({"code": 400, "msg": "query 不能为空"}), 400

    if isinstance(law_ids, str):
        law_ids = [item.strip() for item in law_ids.split(",") if item.strip()]
    law_id_set = {str(item).strip() for item in law_ids if str(item).strip()} or None

    try:
        try:
            from rag.service import run_rag_query

            payload = run_rag_query(
                query,
                top_k=max(1, min(top_k, 15)),
                graph_expand_k=max(0, min(graph_expand_k, 6)),
                law_ids=law_id_set,
                compress=compress,
            )
        except ModuleNotFoundError:
            venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
            if not venv_python.exists():
                raise
            proc = subprocess.run(
                [str(venv_python), "-m", "rag.api"],
                input=json.dumps(
                    {
                        "query": query,
                        "top_k": max(1, min(top_k, 15)),
                        "graph_expand_k": max(0, min(graph_expand_k, 6)),
                        "law_ids": sorted(law_id_set) if law_id_set else [],
                        "compress": compress,
                    },
                    ensure_ascii=False,
                ),
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
        return jsonify({"code": 200, "msg": "ok", "data": payload})
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc))[:400]
        return jsonify({"code": 500, "msg": f"RAG 查询失败: {err}"}), 500
    except Exception as exc:
        return jsonify({"code": 500, "msg": f"RAG 查询失败: {exc}"}), 500


@app.route("/api/agent/query", methods=["POST"])
def agent_query():
    """
    DeepSeek Function-Calling Agent endpoint.

    Request body (JSON)
    -------------------
    {
        "question": "股权善意取得的构成要件是什么？",
        "verbose": false          // optional, default false
    }

    Response body (JSON)
    --------------------
    {
        "code": 200,
        "msg": "ok",
        "data": {
            "answer": "...",
            "rounds": 3,
            "tool_calls": [
                {"round": 1, "tool": "hybrid_search", "args": "...", "output_preview": "..."},
                ...
            ]
        }
    }
    """
    data = request.json or {}
    question = (data.get("question") or "").strip()
    verbose = bool(data.get("verbose", False))
    async_requested = bool(data.get("async", False) or data.get("background", False))
    sync_requested = bool(data.get("sync", False))

    if not question:
        return jsonify({"code": 400, "msg": "question 不能为空"}), 400

    if async_requested and not sync_requested:
        job_id = _start_agent_job(question, verbose=verbose)
        return (
            jsonify(
                {
                    "code": 202,
                    "msg": "accepted",
                    "data": {"job_id": job_id, "status": "queued"},
                }
            ),
            202,
        )

    try:
        result = _run_agent_query_sync(question, verbose=verbose)

        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "answer": result.get("answer", ""),
                    "rounds": result.get("rounds", 0),
                    "tool_calls": result.get("tool_calls", []),
                },
            }
        )

    except subprocess.TimeoutExpired:
        return jsonify({"code": 504, "msg": "Agent 查询超时（超过300秒）"}), 504
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc))[:400]
        return jsonify({"code": 500, "msg": f"Agent 查询失败: {err}"}), 500
    except Exception as exc:
        logger.exception("agent_query failed")
        return jsonify({"code": 500, "msg": f"Agent 查询失败: {exc}"}), 500


@app.route("/api/agent/job/<job_id>", methods=["GET"])
def agent_job_status(job_id):
    _trim_agent_jobs()
    payload = _agent_job_snapshot(job_id)
    if not payload:
        return jsonify({"code": 404, "msg": "Agent 任务不存在或已过期"}), 404
    return jsonify({"code": 200, "msg": "ok", "data": payload})


# ============================================================
# 静态文件伺服（本地开发用；Vercel 部署时由 vercel.json 接管）
# ============================================================

_ROOT = Path(__file__).resolve().parent


@app.route("/")
def index():
    return send_from_directory(str(_ROOT), "index.html")


@app.route("/<path:path>")
def static_files(path):
    """
    Serve any file that physically exists under the project root.
    API routes registered above are matched first, so this only
    handles actual static assets (HTML, JS, CSS, JSON data, images…).
    """
    target = _ROOT / path
    if target.is_file():
        mime, _ = mimetypes.guess_type(str(target))
        return send_from_directory(
            str(_ROOT), path, mimetype=mime or "application/octet-stream"
        )
    # Fall back to index.html for SPA-style navigation
    return send_from_directory(str(_ROOT), "index.html")


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    print("=========================================")
    print(" 法考法条库 后端认证服务器 v2.1")
    print(f" 运行在端口: {port}")
    print(" 激活码算法：HMAC-SHA256 / 10位自校验")
    print("=========================================")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
