# Render Free + Qdrant Cloud Free 部署说明

目标：保留完整 RAG 功能，把后端放到 Render Free，把完整向量库放到 Qdrant Cloud Free。

这个方案不删功能：

- 向量检索：Qdrant Cloud
- BM25：Render 后端内存中按需构建
- 引用图扩展：使用 `js/backlinks.js`
- LightRAG：部署包保留 `rag/storage_lightrag`
- Agent：Render 后端调用 DeepSeek + 工具链

## 为什么这样拆

本地完整 Qdrant 索引在 `rag/storage_full`，约 355MB。Render Free 可以跑后端，但不适合把本地向量库作为持久文件存进去。把向量库迁到 Qdrant Cloud 后，Render 部署包会排除 `rag/storage_full/`，部署更轻，运行时仍然保留完整向量检索。

## 1. 创建 Qdrant Cloud Free

创建一个 Free Cluster，然后拿到：

- `QDRANT_URL`
- `QDRANT_API_KEY`

集合名统一使用：

```bash
law_articles_qwen4b_full
```

## 2. 从本地 Qdrant 迁移到 Qdrant Cloud

先在本机设置环境变量：

```bash
export QDRANT_URL="https://你的-qdrant-cloud-url"
export QDRANT_API_KEY="你的-qdrant-api-key"
export QDRANT_COLLECTION="law_articles_qwen4b_full"
```

然后运行迁移脚本：

```bash
python -m rag.migrate_qdrant_to_cloud --recreate
```

这个脚本会从本地：

```bash
rag/storage_full/qdrant_local
```

读取 `law_articles_qwen4b_full` collection，并直接 upsert 到 Qdrant Cloud。它不会重新调用 SiliconFlow，也不会重新生成 embedding。

迁移完成后可以查：

```bash
python - <<'PY'
from qdrant_client import QdrantClient
import os
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
print(c.get_collection(os.environ.get("QDRANT_COLLECTION", "law_articles_qwen4b_full")))
PY
```

## 3. 部署 Render Free

仓库里已经有：

- `Dockerfile`
- `render.yaml`
- `.dockerignore`

`.dockerignore` 会排除 `rag/storage_full/`，因为向量库已经放到 Qdrant Cloud；但会保留 `rag/storage_lightrag/`。

Render 里创建 Web Service，选择 GitHub 仓库，使用 Blueprint 或 Docker 部署均可。

Render 环境变量至少填写：

```bash
LAW_ACTIVATION_SECRET_KEY=换成一个长随机字符串
QDRANT_URL=https://你的-qdrant-cloud-url
QDRANT_API_KEY=你的-qdrant-api-key
QDRANT_COLLECTION=law_articles_qwen4b_full
SILICONFLOW_API_KEY=你的-siliconflow-key
DEEPSEEK_API_KEY=你的-deepseek-key
```

其他变量 `render.yaml` 已给默认值。

## 4. 前端连接 Render 后端

如果前端和后端同源部署，不用额外设置。

如果前端仍放 Vercel，后端放 Render，需要在页面加载前设置：

```html
<script>
  window.APP_BACKEND_BASE_URL = "https://你的-render-service.onrender.com";
</script>
```

当前 `annotation.html` 和 `js/ai_sidebar.js` 都会优先读取 `window.APP_BACKEND_BASE_URL`。

## 5. 上线检查

Render 部署成功后访问：

```bash
curl https://你的-render-service.onrender.com/api/health
curl https://你的-render-service.onrender.com/api/rag/health
```

期望：

- `/api/health` 返回 `code=200`
- `/api/rag/health` 里：
  - `vector_store=remote`
  - `qdrant_collection=law_articles_qwen4b_full`
  - `has_siliconflow_key=true`
  - `has_chat_key=true`

然后在页面里问一个短问题，例如：

```text
民法典中自然人民事权利能力从什么时候开始？
```

## 注意

Render Free 会休眠，首次访问可能慢。当前认证数据默认写在 `/tmp/users.json` 和 `/tmp/codes.json`，Render Free 重启后可能丢失；如果之后要恢复严格登录/激活，建议另接外部数据库或升级 Render 持久磁盘。
