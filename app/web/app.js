const $ = (id) => document.getElementById(id);
const upload = $("upload");
const chat = $("chat");
const messages = $("messages");
const welcome = $("welcome");
const station = $("station");
const question = $("question");
const importStatus = $("import-status");
const ask = chat.querySelector("button");
let sessionId = null;

function element(tag, text, className) {
  const item = document.createElement(tag);
  item.textContent = text;
  if (className) item.className = className;
  return item;
}

function sourceLink(text, url, preview = false) {
  const link = element("a", text);
  link.href = url;
  if (preview) {
    link.target = "_blank";
    link.rel = "noopener";
  }
  return link;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : `请求失败（${response.status}）`);
  }
  return data;
}

async function refreshLibrary() {
  const manuals = await api("/api/documents");
  const scope = station.value;
  const registered = [...new Set(manuals.map((manual) => manual.station).filter(Boolean))];
  station.replaceChildren(new Option("全部工位", ""), ...registered.map((name) => new Option(name, name)));
  if (registered.includes(scope)) station.value = scope;
  $("count").textContent = `${manuals.length} 份`;
  $("welcome-hint").textContent = manuals.length
    ? "选择检索范围，再输入工位、设备或操作名称。"
    : "导入作业指导书后，输入工位、设备或操作名称进行查询。";
  $("manuals").replaceChildren(...manuals.map((manual) => {
    const row = element("li", "");
    const pages = manual.text_pages === manual.pages
      ? `${manual.pages} 页`
      : `${manual.pages} 页 · ${manual.text_pages} 页有可检索文字`;
    const location = manual.sheets ? `${manual.sheets} 张工作表可检索` : pages;
    const url = `/api/documents/${manual.id}/${manual.sheets ? "original" : "pdf"}`;
    row.append(
      sourceLink(manual.source, url, !manual.sheets),
      element("small", `${manual.station || "未标注工位"} · ${location}`),
    );
    return row;
  }));
}

function report(text, error = false) {
  importStatus.textContent = text;
  importStatus.dataset.error = String(error);
}

async function runImport(operation) {
  const submit = upload.querySelector("button");
  submit.disabled = $("scan").disabled = true;
  report("正在处理指导书…");
  try {
    await operation();
    await refreshLibrary();
  } catch (error) {
    report(error.message, true);
  } finally {
    submit.disabled = $("scan").disabled = false;
  }
}

upload.addEventListener("submit", (event) => {
  event.preventDefault();
  const body = new FormData(upload);
  runImport(async () => {
    const result = await api("/api/documents/upload", { method: "POST", body });
    if (result.status === "unchanged") {
      report("该文件已在知识库中。");
    } else {
      const detail = result.sheets ? `${result.sheets} 张工作表` : `${result.text_pages}/${result.pages} 页`;
      report(`已导入 ${result.source}，${detail}可检索。`);
    }
    $("file").value = "";
  });
});

$("scan").addEventListener("click", () => runImport(async () => {
  const { results } = await api("/api/documents/scan", { method: "POST" });
  if (!results.length) {
    report("dataset 中没有找到 Excel 或 PDF 指导书。");
    return;
  }
  const added = results.filter((item) => item.status === "ready");
  const unchanged = results.filter((item) => item.status === "unchanged");
  const failed = results.filter((item) => item.status === "error");
  const details = failed.map((item) => `${item.source}：${item.detail}`);
  report([`已导入 ${added.length} 份，已存在 ${unchanged.length} 份，失败 ${failed.length} 份。`, ...details].join("\n"), failed.length > 0);
}));

function appendMessage(text, kind = "assistant") {
  welcome.remove();
  const message = element("article", "", `message ${kind}`);
  message.setAttribute("aria-label", kind === "user" ? "你的问题" : "助手回答");
  message.append(element("p", text));
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

function addCitations(message, citations) {
  if (!citations.length) return;
  const list = element("ul", "", "sources");
  list.setAttribute("aria-label", "指导书引用");
  for (const citation of citations) {
    const row = element("li", "");
    const location = citation.sheet ? `${citation.sheet} · ${citation.cell_range}` : `第 ${citation.page} 页`;
    const label = `[${citation.index}] ${citation.source} · ${location}`;
    row.append(sourceLink(label, citation.pdf_url || citation.file_url, Boolean(citation.pdf_url)));
    if (citation.station) row.append(element("p", citation.station));
    row.append(element("p", citation.snippet));
    list.append(row);
  }
  message.append(list);
}

chat.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = question.value.trim();
  if (!query || ask.disabled) return;
  const scope = station.value;
  ask.disabled = $("reset").disabled = station.disabled = true;
  question.value = "";
  appendMessage(scope ? `${scope}\n${query}` : query, "user");
  const reply = appendMessage("正在查找作业指导书…");
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, station: scope, session_id: sessionId, stream: false }),
    });
    sessionId = result.session_id;
    reply.firstElementChild.textContent = result.answer;
    addCitations(reply, result.citations);
    messages.scrollTop = reply.offsetTop - 12;
  } catch (error) {
    reply.firstElementChild.textContent = `查询失败：${error.message}`;
    if (!question.value) question.value = query;
  } finally {
    ask.disabled = $("reset").disabled = station.disabled = false;
  }
});

$("reset").addEventListener("click", () => {
  sessionId = null;
  messages.replaceChildren(welcome);
  question.value = "";
  question.focus();
});

api("/health")
  .then((status) => {
    const vectors = status.vectors;
    const semantic = vectors?.indexed > 0;
    const search = semantic ? "混合检索" : "关键词检索";
    if (status.model_provider === "deepseek") {
      $("mode").textContent = status.model_configured ? `${search} · DeepSeek` : `${search} · DeepSeek 待配置`;
    } else {
      $("mode").textContent = status.model_configured ? `${search} · 问答` : search;
    }
    if (vectors?.model) {
      $("mode").title = `向量覆盖 ${vectors.indexed}/${vectors.chunks} 个片段`;
    }
  })
  .catch(() => { $("mode").textContent = "服务连接失败"; });
api("/api/access")
  .then((access) => { $("import-controls").hidden = !access.can_import; })
  .catch((error) => report(error.message, true));
refreshLibrary().catch((error) => report(error.message, true));
