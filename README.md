# 知源 · 产线作业助手

使用 Excel 和 PDF 作业指导书查询工位操作、检查要求和注意事项。PDF 引用可打开对应页，Excel 引用会标出工作表及单元格范围，并提供原文件下载。

## Windows 本机运行

需要 Python 3.10 或更新版本。在项目目录执行以下命令。已有可用的 Python 时也可以把 `py -3` 换成 `python`。

```powershell
py -3 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv-local\Scripts\python.exe -m app.main
```

打开 [本地界面](http://127.0.0.1:8000)。

## 导入作业指导书

- 在页面上传 `.xlsx`、`.xlsm` 或 `.pdf`，并按需填写所属工位。选择指定工位后，查询只检索该工位下的指导书。
- 也可以把 Excel 和 PDF 一起放进 `dataset`，再点「扫描 dataset 文件夹」。例如 `dataset/A01 装配/装配作业指导书.xlsx` 会归入 `A01 装配` 工位。直接放在 `dataset` 下的文件会从文件名识别 `WS1001` 这类工位号。提问时可在问题中写明工位号。覆盖同工位、同文件名的新版文件后，再次导入即可更新检索内容。
- 旧版 `.xls` 请先另存为 `.xlsx`。Excel 检索读取可见工作表的单元格内容和公式已保存的结果。引用与嵌入图片的图例区域匹配时，界面显示原图预览，可点击展开。矢量绘图和附件可在原文件中查看。
- 请使用可选中文字的 PDF。纯扫描件需要先做 OCR。含图纸或图片步骤的指导书，可以通过页码引用查看原图。

原文件和检索索引保存在 `data/instructions.sqlite3`。新版入库后，旧版的引用链接仍能打开当时的原文。

## 启用本地 RAG

在项目根目录的 `.env` 中设置 `DEEPSEEK_API_KEY`，然后在 PowerShell 执行 `.\scripts\start-local.ps1`。如果应用已启动，使用 `.\scripts\start-local.ps1 -Restart` 重新加载配置。也可以在启动服务的终端中设置同名环境变量。密钥仅供后端读取。

1. 安装 `requirements-rag.txt`。将 `.env.example` 复制为 `.env`，启用 `ZHIYUAN_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`。有本地 ONNX 文件时，可以通过 `ZHIYUAN_RAG_EMBEDDING_PATH` 指定其目录。首次使用且未指定目录时，FastEmbed 会下载模型。
2. 运行 `python -m app.index_vectors`，为已经入库的文档补建向量。脚本可以重复运行，会跳过已有的当前模型向量。改用其他嵌入模型或同名模型的新权重时，更新模型名或 `ZHIYUAN_RAG_EMBEDDING_REVISION`，再次运行此命令。
3. DeepSeek 配置使用 `https://api.deepseek.com` 和 `deepseek-flash`。DeepSeek 先根据当前问题和近期对话规划检索，再依据命中的指导书片段组织回答。问题、近期对话和命中的原文片段会发送给 DeepSeek，原始 Excel、PDF 和向量库保存在本机。

新文件入库时，文本、出处及向量一起提交。检索先按工位限定范围，再用 SQLite FTS5 和向量余弦相似度分别召回，经 RRF 融合排序。可选 `ZHIYUAN_RAG_RERANKER_MODEL=BAAI/bge-reranker-base` 对候选片段进一步重排。`/health` 返回当前向量模型和覆盖数量。问答模型不可用时，界面仍可显示检索到的原文。

DeepSeek 回答会附上支持该段回答的引文。服务核对引文能否在对应片段中找到，再由应用生成 PDF 页码或 Excel 单元格引用。密钥缺失或调用失败时，仍可使用本机检索。

## 内网部署

- 默认监听 `127.0.0.1`。设置 `ZHIYUAN_STAFF_PASSWORD` 和 `ZHIYUAN_ADMIN_PASSWORD` 后可分别用 `staff` 查询和 `admin` 管理导入。通过 HTTPS 反向代理提供给其他机器，保留外部 Host 和转发协议。请求日志记录方法、路由、状态、耗时和请求 ID。
- 当前向量保存在同一 SQLite 库中，语义检索在工位范围内执行精确余弦搜索。部署到多个节点或资料量显著增长前，应使用实际语料评估吞吐和召回，再决定是否迁移到 Milvus 等独立向量库。共享账号适用于试点，逐人审计或工位级访问授权应接入企业身份系统。

### 单机并发与故障处理

- 查询接口共用本机准入上限，默认同时接收 32 个查询，每个工位最多 4 个。超过上限返回 HTTP 429 和 `Retry-After`。导入共用一个单独的执行名额，SQLite 使用 WAL。每份指导书解析和向量化完成后，再一次性更新该份文件与索引。会话步骤位置保存在独立的 `instructions.sessions.sqlite3`，避免导入写锁阻塞问答保存步骤。
- DeepSeek 共用连接池，默认最多 8 个并发调用，排队最多 0.2 秒。一次问答内的多个模型阶段共用从问答开始计时的 8 秒预算。本机检索和原文回退可以继续执行。可用 `ZHIYUAN_LLM_REQUESTS_PER_MINUTE` 设置滚动一分钟调用上限，默认 0 表示未配置分钟配额。
- 上游 429、401、403 会立即熔断，连续 3 次网络错误、超时或 5xx 也会熔断。默认冷却 15 秒，遇到以秒表示的更长 `Retry-After` 会延长冷却。冷却后只放行一个模型探测，成功后恢复。熔断、模型拥堵或失败时，问答可返回原文出处，已核对的相邻步骤也可直接返回。
- 在 `/health` 的 `load` 中查看查询、导入和模型的在途数量、拒绝计数、熔断状态及回答结果。阈值在 `.env.example` 中列出，按单个服务进程统计。多进程或多实例部署时，需要把跨进程限流和熔断移到共享网关或协调层。

运行 `python -m scripts.benchmark_resilience` 可用临时合成指导书模拟换班流量、导入并行、上游超时和 429。脚本在进程内走 HTTP 应用接口，用模拟模型替换外部调用。
