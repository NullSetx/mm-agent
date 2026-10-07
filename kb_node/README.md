# kb_node · 百科知识库节点

`kb_search` 一个工具。跑在 **16GB 节点（A）**上，纯 CPU、无 GPU 依赖。
方案与选型依据（向量库 / embedding 对比、数据流）：[docs/知识库方案.md](../docs/知识库方案.md)。

| 工具 | 依赖 | 用途 |
|---|---|---|
| `kb_search` | embedding API（默认 SiliconFlow `BAAI/bge-m3`）+ Chroma | 检索 `wiki/` 百科词条与 `docs/` 项目文档，返回带来源的相关片段——Agent 靠它回答「XX 是什么 / 原理」这类百科问题和「怎么启动 / 谁负责什么」这类项目问题 |

## 语料

| 目录 | 内容 | 谁维护 |
|---|---|---|
| `wiki/` | **百科词条**（`.md`/`.txt`，一个文件一个词条，markdown 标题分章节） | 你按需增删 |
| `docs/` | 项目文档（接口约定、方案等） | 三人共同 |

词条格式没有特殊要求：普通 markdown，标题层级会被切成带「文件名 › 章节」
前缀的检索块（`wiki/` 下放两个示例词条可参考）。**加词条 = 丢个 .md 进
`wiki/`，重跑一次入库**。

### 只收录 / 只检索某个目录

**入库路径 = 语料范围**。默认收 `wiki/` + `docs/` + 根 README，显式给路径
则整体替换（上次收过、这次不在路径里的文件，其旧块会被清掉）：

```bash
python -m kb_node.ingest wiki              # 库里只留 wiki/
python -m kb_node.ingest D:\我的百科       # 任意目录（相对/绝对均可）
python -m kb_node.ingest wiki docs         # 多个目录全列上，缺了就清
```

想长期改默认目录集合，改 `kb_node/ingest.py` 的默认 paths 一行。
查询侧永远不用配——库里有什么就检索什么。

> 想"一个库里多套语料、查询时按问题只搜其中一个"（如问百科只搜 `wiki/`、
> 问项目只搜 `docs/`），目前不支持，属候选增强，见
> [docs/分工与接口约定.md §9.2](../docs/分工与接口约定.md)。

## 启动

```bash
# 从仓库根目录
pip install -r requirements.txt        # chromadb 在 kb_node 分区，A 的机器装

# 配 embedding key（写仓库根目录 .env 或环境变量，绝不进仓库；SiliconFlow 免费。
# .env 由 common/config.py 自动加载，优先级：显式环境变量 > .env；模板见 .env.example）
KB_EMBEDDING_API_KEY=sk-...

# 入库（默认增量，改完语料重跑一次即可）+ 起节点
python -m kb_node.ingest
uvicorn kb_node.server:app --host 0.0.0.0 --port 8103

# 没有 key 时，mock 模式先把链路跑通（文档 §6.2，不需要 chromadb）
NODE_MOCK=1 uvicorn kb_node.server:app --host 0.0.0.0 --port 8103
```

入库完成后重启一次网关（或 `POST /api/tools/refresh`），`kb_search` 就会
出现在 `GET /api/tools` 里，Agent 自动多出这个工具——网关代码零改动。

## 接口说明

遵循 `docs/分工与接口约定.md` §4.2，对外只有三个端点。**网关（A）对接只需要看这一节。**

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活 + 已注册工具名，网关的 `/api/health` 聚合这个 |
| GET | `/tools` | 自报工具清单，网关照这个发现工具 |
| POST | `/invoke` | 统一调用入口 |

### GET /health

```json
{
  "node": "kb",
  "status": "ok",
  "tools": ["kb_search"],
  "detail": {"chunks": 44, "embedding": "api:BAAI/bge-m3"}
}
```

`detail` 里的 `chunks` 是已收录块数、`embedding` 是入库时的向量模型签名；
未入库 / 未装 chromadb 时给空态或错误说明，**网关不依赖 detail 内容**。

### POST /invoke

```json
// 请求（needs_image=false，不收图片）
{"tool": "kb_search", "params": {"query": "怎么刷新工具清单", "topk": 3}}

// 成功响应的 result
{
  "query": "怎么刷新工具清单",
  "hits": [
    {"text": "docs/分工与接口约定.md › §5\n调一次 POST /api/tools/refresh……",
     "source": "docs/分工与接口约定.md › §5", "score": 0.83}
  ],
  "total": 3
}
```

- `query` 必填（空值报 `ValueError` → `ok=false`）；`topk` 默认 3，上限 10；
- `score` 是 cosine 相似度（1 - 距离），仅供参考，Agent 的观察文本里不展示；
- embedding API 不可用时 `ok=false`（HTTP 仍 200），对话不崩。

## 入库与更新

**更新 = 改 `wiki/` 或 `docs/*.md` → 重跑 `python -m kb_node.ingest`**，默认
**增量**：按内容指纹（sha256）跳过没变的文件，只重嵌改动过的，词条量大了
也不怕慢、不浪费 API 钱：

- 新增 / 修改 / 删除词条后重跑即可；被删文件的旧块自动清掉，文件内容没变
  就一块都不重嵌（报告里分别列 `新嵌 / 跳过` 篇数）；
- ⚠️ **必须重启 kb 节点**（2026-10-07 实测更正，原文这条写反了）。
  旧文档说"节点**不用重启**，每次查询都重新打开集合，重入库后下一次查询即生效"，
  **是错的**：入库跑在**另一个进程**里，而节点内缓存的 `PersistentClient`
  （`store.py:51`）内存中那份 HNSW 索引，不会因为别的进程写了盘就重载。
  实测同一句 query：跑着的节点返回**旧文件**（最高分 0.34，新文件一块没出来）；
  全新进程和**重启后的节点**都命中新文件（0.67）。
  所以流程是 **改语料 → `ingest` → 重启 kb 节点**；
- 工具描述里的「收录 N 个片段」同样是节点启动时算的；重启刷新节点后，网关侧还要
  `POST /api/tools/refresh`（或重启网关）才看得到新数字——这一步只影响模型看到的
  覆盖面提示，不影响检索；
- `--rebuild` 强制全量重嵌（换 embedding 模型后不需要——检测到配置变化会
  自动清库重建；结果可疑或中途断过才用）；
- **入库与查询必须是同一 embedding 配置**（provider+model），配置写在集合
  metadata 里，换模型后查询旧库会直接报错提示重新入库——两套向量空间不通用；
- 切换 embedding 供应商：改 `KB_EMBEDDING_*` 环境变量后重跑入库即可
  （默认 SiliconFlow bge-m3；智谱 `embedding-3` 同为 OpenAI 兼容，改 3 个变量）；
- 断网兜底：`KB_EMBEDDING_PROVIDER=local`（需装 sentence-transformers，可选依赖），
  演示日网络不稳时提前用本地模型重入库。

## 设计要点

- **块带「文档名 › 章节路径」前缀入库**：前缀参与检索、`source` 单独存
  元数据，回答能标注出处；Agent 侧渲染成带来源的权威观察文本，未命中
  明确说「没有相关内容」，杜绝编造；
- **chromadb 懒加载**：未安装时节点照常起、`/health` 照常绿，`kb_search`
  返回可读错误——mock-first，不挡三方联调；
- **入库默认增量、走 CLI 不走 LLM 工具**：写库不该由模型决策，模型只管检索；
  指纹表（manifest.json）跟 Chroma 一起存在 data/kb/ 里。

## 测试

```bash
pytest tests/test_kb.py     # 切分器 / embedding 客户端（不出网）/ 入库检索 / 网关发现
pytest tests/               # 全量 47 项
```

真实 Chroma 的用例在未安装 chromadb 时自动跳过；embedding 用
`httpx.MockTransport` 假传输层，测试不出网。
