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
- 旧版 `.xls` 请先另存为 `.xlsx`。Excel 检索读取可见工作表的单元格内容和公式已保存的结果。图片、绘图和附件可在原文件中查看。
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
