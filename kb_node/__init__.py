"""kb_node：项目知识库节点（方案与选型见 docs/知识库方案.md）。

把 docs/ 项目文档切分、向量化入 Chroma，经 `kb_search` 工具供 Agent
检索——回答「怎么启动 / 谁负责什么 / 接口参数」这类项目自身的问题。

纯 CPU、无 GPU 依赖；embedding 走 OpenAI 兼容 API（默认 SiliconFlow
bge-m3），断网可切本地 sentence-transformers 兜底。
"""
