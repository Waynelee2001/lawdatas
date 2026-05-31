# 法考法条库

法考法条库是一个面向法考学习和法律条文查阅的轻量级法规检索站点。项目以静态前端为主，内置法律法规正文、标注版正文、法规目录树、交叉引用索引和新旧法条对照数据，并提供一个 Flask 认证服务用于账号注册、登录、兑换码激活和密码修改。

线上默认访问地址：

- https://lawdatas.vercel.app

## 功能特性

- 法律法规目录树：按照法考常用分类组织法律、司法解释和相关规范文件。
- 普通版与标注版：`index.html` 提供基础阅读视图，`annotation.html` 提供标注、引用和学习增强视图。
- 法条检索与定位：支持目录搜索、正文搜索、上一条/下一条匹配跳转、侧边栏展开收起和拖拽调整宽度。
- 法条交叉引用：识别正文中的其他法条引用，支持悬浮预览和点击跳转。
- 反向引用：通过 `js/backlinks.js` 展示当前条文被其他条文引用的来源。
- 法条收藏：标注版中可收藏具体条文，并在 `favorites.html` 中按分类集中查看。
- 新旧法条对照：通过 `compare.html?id=<lawId>` 查看重点法律的新旧条号对照表。
- 账号与激活：后端提供注册、登录、兑换码激活、修改密码接口；本地开发默认连接 `http://127.0.0.1:5100`。

## 项目结构

```text
.
├── index.html                  # 普通版法条库入口
├── annotation.html             # 标注版入口，包含收藏、交叉引用、账号弹窗等增强功能
├── favorites.html              # 收藏法条视图
├── compare.html                # 新旧法条对照表视图
├── auth_server.py              # Flask 认证与兑换码服务
├── keygen_tool.py              # 兑换码生成与校验工具
├── generate_backlinks.js       # 从标注数据生成反向引用索引
├── all_laws_map.json           # 法律 ID 与法律名称映射
├── data/
│   ├── laws/                   # 法规正文数据
│   ├── laws_annotation/        # 标注版法规正文数据
│   ├── annotations/            # 独立标注数据
│   └── compare_*.json          # 新旧法条对照数据
├── js/
│   ├── tree_data.js            # zTree 目录数据
│   ├── law_index.js            # 法律名称到 ID 的索引
│   └── backlinks.js            # 反向引用索引
├── css/                        # zTree 样式
├── img/                        # 图标、二维码等静态资源
├── requirements.txt            # Python 后端依赖
├── vercel.json                 # Vercel 部署配置
└── Procfile                    # Gunicorn 启动配置
```

当前仓库内置数据概览：

- `data/laws/`：287 份法规正文 JSON
- `data/laws_annotation/`：287 份标注版法规正文 JSON
- `data/annotations/`：157 份独立标注 JSON
- `all_laws_map.json`：405 条法律 ID/名称映射
- `data/compare_*.json`：2 份新旧法条对照数据

## 本地运行

### 1. 启动静态前端

浏览器直接打开 HTML 文件时，部分浏览器会限制本地 JSON 请求。建议在项目根目录启动一个本地静态服务：

```bash
python3 -m http.server 8000
```

然后访问：

- 普通版：http://127.0.0.1:8000/index.html
- 标注版：http://127.0.0.1:8000/annotation.html
- 收藏页：http://127.0.0.1:8000/favorites.html

### 2. 启动认证服务

标注版在本地环境会把账号相关请求发送到 `http://127.0.0.1:5100`。如需测试注册、登录和兑换码功能，另开一个终端运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python auth_server.py
```

服务启动后会监听 `0.0.0.0:5100`，提供以下接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/register` | 注册账号 |
| `POST` | `/login` | 登录账号 |
| `POST` | `/redeem` | 兑换激活码 |
| `POST` | `/change_password` | 修改密码 |

## 常用脚本

生成反向引用索引：

```bash
node generate_backlinks.js
```

生成兑换码：

```bash
python3 keygen_tool.py
python3 keygen_tool.py -n 50 -o codes.txt
```

校验兑换码：

```bash
python3 keygen_tool.py --verify ABCDE-FGHJK
```

## 部署

项目已包含 `vercel.json`，适合部署到 Vercel：

- `auth_server.py` 使用 `@vercel/python` 作为 Serverless 后端。
- HTML、JS、CSS、JSON、图片和 xlsx 文件作为静态资源输出。
- `/register`、`/login`、`/redeem`、`/change_password` 路由会转发到 Python 后端。
- 其他静态路由优先走文件系统，兜底返回 `index.html`。

当前后端为了适配 Vercel Serverless，将轻量用户数据写入 `/tmp/users.json` 和 `/tmp/codes.json`。该方式适合演示和短期测试，不适合作为生产环境的持久化账号系统。正式使用时建议迁移到数据库，并将认证密钥等敏感配置改为环境变量管理。

## 数据格式简述

法规正文数据示例结构：

```json
{
  "id": "2071",
  "name": "人民检察院刑事诉讼规则",
  "content": [
    {
      "lawWebContent": "第一章 通则...",
      "id": "2078",
      "position": 2111,
      "sort": 0,
      "type": "3",
      "content": "第一章 通则",
      "parentId": "2071"
    }
  ]
}
```

新旧法条对照数据示例结构：

```json
{
  "title": "中华人民共和国民事诉讼法 新旧法条对照表",
  "headers": ["1991年文本", "2007年修正文本", "2012年修正文本"],
  "rows": [["第1条", "第1条", "第1条"]]
}
```

## 维护建议

- 新增法规正文后，同步更新 `data/laws/`、`data/laws_annotation/`、`all_laws_map.json` 和 `js/law_index.js`。
- 调整标注正文或引用识别规则后，重新运行 `node generate_backlinks.js`。
- 新增新旧法条对照时，按 `data/compare_<lawId>.json` 命名，并在前端需要展示的位置补充对应 lawId。
- 生产环境不要依赖 `/tmp` 文件存储用户和兑换码状态。

## 许可证

本仓库暂未声明开源许可证。如需复用代码或数据，请先联系作者确认授权范围。
