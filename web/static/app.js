/* mm-agent 演示前端逻辑：只调 /api/*（同源代理 → 网关），模型调用全部走 HTTP 端口。 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  target: localStorage.getItem("gwTarget") || "",   // 空的则用服务端默认（GATEWAY_URL / common.config）
  image: null,                                       // data:image/jpeg;base64,...（对话与工具共用）
  imageName: "",
  tools: [],
  lastBoxes: null,                                   // 最近一次 detect 的框，缩放时重画用
  session: localStorage.getItem("gwSession") || newSessionId(),
  busy: false,
};
localStorage.setItem("gwSession", state.session);

function newSessionId() {
  return "demo-" + Math.random().toString(36).slice(2, 10);
}

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/* ---------------------------------------------------------------- 请求 */

async function api(path, { method = "GET", body } = {}) {
  const headers = {};
  if (state.target) headers["X-Gateway-Target"] = state.target;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await resp.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail
      : (typeof data === "string" && data ? data : `HTTP ${resp.status}`);
    throw new Error(`${detail}`);
  }
  return data;
}

function setGwState(kind, text) {
  const pill = $("#gwState");
  pill.className = "pill " + kind;
  pill.textContent = text;
}

/* ---------------------------------------------------------------- 网关地址 */

async function loadConfig() {
  try {
    const cfg = await api("/__config");
    $("#gwTarget").placeholder = cfg.fallback_gateway;
    if (!state.target) $("#gwTarget").value = "";
    $("#imgInfo").textContent = state.image ? state.imageName : "未选择";
    return cfg;
  } catch (err) {
    $("#statusHint").textContent = "读取本服务配置失败：" + err.message;
    return null;
  }
}

function applyTarget() {
  const raw = $("#gwTarget").value.trim();
  state.target = raw;
  if (raw) localStorage.setItem("gwTarget", raw);
  else localStorage.removeItem("gwTarget");
  refreshHealth();
}

/* ---------------------------------------------------------------- 状态总览 */

function nodeCard(key, title, node) {
  const ok = node && node.ok;
  const tools = (node && node.tools) || [];
  return `<div class="node ${ok ? "ok" : "bad"}">
    <div class="name"><span class="dot"></span>${esc(title)}</div>
    <div class="meta">${ok ? "在线" : "不可用"}${node && node.elapsed_ms != null ? " · " + node.elapsed_ms + " ms" : ""}</div>
    ${tools.length ? `<div class="tools">${tools.map((t) => `<span>${esc(t)}</span>`).join("")}</div>` : ""}
    ${node && node.error ? `<div class="meta">${esc(node.error)}</div>` : ""}
  </div>`;
}

async function refreshHealth() {
  const grid = $("#statusGrid");
  grid.innerHTML = `<div class="node"><div class="name"><span class="dot"></span>正在探测…</div></div>`;
  try {
    const h = await api("/api/health");
    const gwTools = (h.gateway && h.gateway.tools) || [];
    const vllm = h.vllm || { ok: false };
    const cards = [
      nodeCard("gateway", "网关 :8000", { ok: true, tools: gwTools, elapsed_ms: null }),
      nodeCard("vllm", "vLLM :8001", { ok: vllm.ok, tools: vllm.models || [], elapsed_ms: vllm.elapsed_ms, error: vllm.error }),
      nodeCard("vision-fast", "vision-fast :8101（B）", h.nodes && h.nodes["vision-fast"]),
      nodeCard("vision-heavy", "vision-heavy :8102（C）", h.nodes && h.nodes["vision-heavy"]),
    ];
    grid.innerHTML = cards.join("");
    setGwState(h.all_ok ? "ok" : "bad", h.all_ok ? "全链路就绪" : "部分节点未就绪");
    $("#statusHint").textContent = h.all_ok
      ? "全绿。可以开始对话或直接调用工具。"
      : "有节点未就绪：确认对应机器已启动节点、网关地址正确（演示时网关在 A 的机器上）。";
    await loadTools();
  } catch (err) {
    grid.innerHTML = `<div class="node bad"><div class="name"><span class="dot"></span>网关不可达</div>
      <div class="meta">${esc(err.message)}</div></div>`;
    setGwState("bad", "网关不可达");
  }
}

async function refreshTools() {
  const hint = $("#statusHint");
  try {
    const r = await api("/api/tools/refresh", { method: "POST", body: {} });
    const parts = Object.entries(r.nodes || {}).map(
      ([name, info]) => `${name}: ${info.ok ? "ok" : "失败(" + (info.error || "") + ")"}`
    );
    hint.textContent = "工具发现结果 → " + parts.join("；");
    await refreshHealth();
  } catch (err) {
    hint.textContent = "重新发现工具失败：" + err.message;
  }
}

/* 工具清单来自 GET /api/tools（带 params，参数表单要用），
   不能用 /api/health 里的工具名数组——那只是名字。 */
async function loadTools() {
  try {
    const data = await api("/api/tools");
    const specs = Array.isArray(data) ? data : (data && data.tools) || [];
    renderToolOptions(specs);
  } catch (err) {
    $("#toolResult").innerHTML = `<p class="hint">读取工具清单失败：${esc(err.message)}</p>`;
  }
}

/* ---------------------------------------------------------------- 图片 */

function drawSampleImage() {
  const canvas = document.createElement("canvas");
  canvas.width = 640; canvas.height = 420;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#dbeafe"; ctx.fillRect(0, 0, 640, 420);
  ctx.fillStyle = "#f59e0b"; ctx.fillRect(40, 60, 200, 300);
  ctx.fillStyle = "#111827"; ctx.fillRect(300, 200, 280, 160);
  ctx.fillStyle = "#ffffff"; ctx.font = "20px system-ui, sans-serif";
  ctx.fillText("sample (synthetic)", 320, 120);
  return canvas.toDataURL("image/jpeg", 0.9);
}

async function compressToDataUrl(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("读取文件失败"));
    reader.readAsDataURL(file);
  });
  const img = await loadImage(dataUrl);
  const maxSide = 1280;
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  if (scale >= 1 && file.size < 900 * 1024) return dataUrl;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片无法解码"));
    img.src = src;
  });
}

async function setImage(dataUrl, name) {
  state.image = dataUrl;
  state.imageName = name;
  state.lastBoxes = null;          // 换图后旧框作废
  const preview = $("#imgPreview");
  preview.src = dataUrl;
  preview.style.display = "block";
  $("#imgPlaceholder").style.display = "none";
  clearOverlay();
  await waitImage(preview);
  $("#imgInfo").textContent = `${name} · ${preview.naturalWidth}×${preview.naturalHeight}`;
}

function waitImage(img) {
  return img.complete ? Promise.resolve() : new Promise((r) => { img.onload = r; });
}

function clearImage() {
  state.image = null;
  state.imageName = "";
  state.lastBoxes = null;
  $("#imgPreview").removeAttribute("src");
  $("#imgPreview").style.display = "none";
  $("#imgPlaceholder").style.display = "inline-block";
  $("#imgInfo").textContent = "未选择";
  $("#fileInput").value = "";
  clearOverlay();
}

/* ---------------------------------------------------------------- detect 叠加框 */

function clearOverlay() {
  const canvas = $("#overlay");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.width = "0px";
  canvas.style.height = "0px";
}

function drawBoxes(boxes) {
  const img = $("#imgPreview");
  const canvas = $("#overlay");
  if (!img.naturalWidth || !boxes || !boxes.length) return;

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const ratio = img.clientWidth / img.naturalWidth;
  const font = Math.max(12, Math.round(16 / (ratio || 1)));

  ctx.lineWidth = Math.max(1.5, 3 / (ratio || 1));
  ctx.font = `${font}px system-ui, sans-serif`;
  boxes.forEach((b) => {
    const [x1, y1, x2, y2] = b.xyxy;
    ctx.strokeStyle = "#ef4444";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${b.label} ${(b.conf * 100).toFixed(0)}%`;
    const w = ctx.measureText(label).width + 8;
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(x1, Math.max(0, y1 - font - 6), w, font + 6);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 4, Math.max(font, y1 - 6));
  });
}

/* ---------------------------------------------------------------- 工具直调 */

function renderToolOptions(specs) {
  state.tools = specs;
  const select = $("#toolSelect");
  const keep = select.value;
  select.innerHTML = specs.map(
    (t) => `<option value="${esc(t.name)}">${esc(t.name)} — ${esc((t.description || "").slice(0, 26))}</option>`
  ).join("");
  if (specs.some((t) => t.name === keep)) select.value = keep;
  renderParamForm();
}

function renderParamForm() {
  const spec = state.tools.find((t) => t.name === $("#toolSelect").value);
  const box = $("#paramForm");
  if (!spec) { box.innerHTML = ""; return; }
  const params = spec.params || {};
  const keys = Object.keys(params);
  if (!keys.length) { box.innerHTML = `<span class="hint">该工具无需额外参数</span>`; return; }
  box.innerHTML = keys.map((key) => {
    const def = params[key];
    const id = "p_" + key;
    let input;
    if (typeof def === "boolean") {
      input = `<input type="checkbox" id="${id}" ${def ? "checked" : ""}>`;
    } else if (typeof def === "number") {
      input = `<input type="number" id="${id}" step="${Number.isInteger(def) ? "1" : "0.01"}" value="${def}">`;
    } else {
      const val = def == null ? "" : (typeof def === "object" ? JSON.stringify(def) : String(def));
      input = `<input type="text" id="${id}" value="${esc(val)}" placeholder="可留空">`;
    }
    return `<label for="${id}">${esc(key)}<span style="display:flex;gap:6px;align-items:center">${input}</span></label>`;
  }).join("");
}

function collectParams() {
  const spec = state.tools.find((t) => t.name === $("#toolSelect").value);
  const out = {};
  if (!spec) return out;
  Object.entries(spec.params || {}).forEach(([key, def]) => {
    const input = $("#p_" + key);
    if (!input) return;
    if (input.type === "checkbox") { out[key] = input.checked; return; }
    const raw = input.value.trim();
    if (raw === "") return;                       // 留空则不传，用工具默认值
    if (typeof def === "number") { out[key] = Number(raw); return; }
    if (def == null || typeof def === "object") {
      try { out[key] = JSON.parse(raw); } catch { out[key] = raw; }
      return;
    }
    out[key] = raw;
  });
  return out;
}

async function invokeSelected() {
  const tool = $("#toolSelect").value;
  if (!tool) { $("#toolResult").innerHTML = `<p class="hint">先选择工具</p>`; return; }
  const spec = state.tools.find((t) => t.name === tool) || {};
  const box = $("#toolResult");
  box.innerHTML = `<p class="hint">调用 ${esc(tool)} 中…</p>`;
  try {
    const payload = { tool, params: collectParams() };
    if (spec.needs_image !== false) {
      if (!state.image) { box.innerHTML = `<p class="hint">该工具需要图片，请先在「当前图片」里选一张。</p>`; return; }
      payload.image = state.image;
    }
    const t0 = performance.now();
    const resp = await api("/api/invoke", { method: "POST", body: payload });
    const wall = Math.round(performance.now() - t0);
    box.innerHTML = renderInvokeResult(tool, resp, wall);
    if (resp.ok && resp.result && resp.result.boxes) {
      state.lastBoxes = resp.result.boxes;
      drawBoxes(resp.result.boxes);
    }
  } catch (err) {
    box.innerHTML = `<h3>调用失败</h3><pre class="json">${esc(err.message)}</pre>`;
  }
}

function renderInvokeResult(tool, resp, wall) {
  if (!resp.ok) {
    return `<h3>${esc(tool)} · 工具返回失败</h3>
      <pre class="json">${esc(resp.error || "未提供错误信息")}</pre>
      <p class="hint">节点内部报错也返回 HTTP 200，用 ok=false 表达——看这里就能确认容错设计生效。</p>`;
  }
  const r = resp.result || {};
  let body = "";

  if (Array.isArray(r.boxes)) {
    const rows = r.boxes.map((b) => `<tr><td>${esc(b.label)}</td><td>${(b.conf * 100).toFixed(1)}%</td>
      <td>${b.xyxy.map((v) => Math.round(v)).join(", ")}</td></tr>`).join("");
    body = `<h3>检出 ${r.count} 个目标（${esc(tool)}）</h3>
      ${r.boxes.length ? `<table class="boxes"><thead><tr><th>类别</th><th>置信度</th><th>xyxy</th></tr></thead><tbody>${rows}</tbody></table>` : `<p class="hint">没有检出目标——纯色/合成图属正常结果。</p>`}
      <p class="hint">图宽高 ${r.width}×${r.height} · 阈值 ${r.conf_threshold} · 节点内推理 ${r.elapsed_s ?? "-"}s</p>`;
  } else if (Array.isArray(r.predictions)) {
    const bars = r.predictions.map((p) => `<div class="bar"><span>${esc(p.label)}</span>
      <span class="track"><span class="fill" style="width:${Math.max(2, p.score * 100)}%"></span></span>
      <span class="val">${(p.score * 100).toFixed(1)}%</span></div>`).join("");
    body = `<h3>top-${r.predictions.length}（${esc(tool)}）</h3><div class="bars">${bars}</div>`;
  } else if (Array.isArray(r.texts)) {
    body = `<h3>识别到 ${r.texts.length} 段文字（${esc(tool)}）</h3>
      <ul>${r.texts.map((t) => `<li>${esc(typeof t === "string" ? t : JSON.stringify(t))}</li>`).join("")}</ul>`;
  } else {
    const imgField = ["image", "image_base64", "output_image", "result_image"].find((k) => typeof r[k] === "string" && r[k].length > 200);
    if (imgField) {
      const src = r[imgField].startsWith("data:") ? r[imgField] : "data:image/png;base64," + r[imgField];
      body = `<h3>返回图片（${esc(tool)}）</h3><img src="${src}" alt="工具返回图" style="max-width:100%;border-radius:8px">`;
    } else {
      body = `<h3>${esc(tool)} 结果</h3>`;
    }
  }

  return `${body}
    <p class="hint">节点耗时 ${resp.elapsed_ms} ms · 端到端 ${wall} ms（含浏览器→代理→网关→节点）</p>
    <details class="raw"><summary>原始 JSON</summary><pre class="json">${esc(JSON.stringify(resp, null, 2))}</pre></details>`;
}

async function selfCheck() {
  const box = $("#toolResult");
  box.innerHTML = `<p class="hint">自检进行中…</p>`;
  const lines = [];
  for (const tool of ["detect", "classify"]) {
    const spec = state.tools.find((t) => t.name === tool);
    if (!spec) { lines.push(`${tool}: 网关未发现该工具（节点没起或未刷新）`); continue; }
    try {
      const payload = { tool, params: {} };
      if (spec.needs_image !== false) payload.image = state.image;
      const t0 = performance.now();
      const resp = await api("/api/invoke", { method: "POST", body: payload });
      const wall = Math.round(performance.now() - t0);
      const extra = tool === "detect" ? `检出 ${resp.result ? resp.result.count : "-"} 个目标`
        : `top1 ${resp.result && resp.result.top1 ? resp.result.top1.label + " " + (resp.result.top1.score * 100).toFixed(1) + "%" : "-"}`;
      lines.push(`${tool}: ${resp.ok ? "ok" : "失败"} · ${extra} · 节点 ${resp.elapsed_ms} ms · 端到端 ${wall} ms`);
      if (tool === "detect" && resp.ok && resp.result && resp.result.boxes) {
        state.lastBoxes = resp.result.boxes;
        drawBoxes(resp.result.boxes);   // 自检也把框画在图上，现场最直观
      }
    } catch (err) {
      lines.push(`${tool}: 调用异常 · ${err.message}`);
    }
  }
  box.innerHTML = `<h3>自检结果</h3><pre class="json">${esc(lines.join("\n"))}</pre>`;
}

/* ---------------------------------------------------------------- 对话 */

function addMessage(role, text, calls) {
  const log = $("#chatLog");
  const div = document.createElement("div");
  div.className = "msg " + role;
  const who = role === "user" ? "我" : role === "err" ? "错误" : "助手";
  let callsHtml = "";
  if (calls && calls.length) {
    callsHtml = `<div class="calls">${calls.map((c) => {
      const ok = c.ok !== false;
      const brief = ok ? summarize(c.result) : (c.error || "失败");
      return `<div class="call ${ok ? "" : "bad"}"><span class="tag">${esc(c.tool)}</span>${esc(brief)}</div>`;
    }).join("")}</div>`;
  }
  div.innerHTML = `<div class="who">${who}</div><div class="bubble">${esc(text)}</div>${callsHtml}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function summarize(result) {
  if (!result || typeof result !== "object") return String(result ?? "");
  if (Array.isArray(result.boxes)) return `检出 ${result.count} 个目标`;
  if (Array.isArray(result.predictions)) return `top1 ${result.predictions[0] ? result.predictions[0].label : "-"}`;
  if (Array.isArray(result.texts)) return `${result.texts.length} 段文字`;
  const keys = Object.keys(result).slice(0, 4).join(", ");
  return `{${keys}}`;
}

async function sendChat() {
  if (state.busy) return;
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;

  state.busy = true;
  $("#btnSend").disabled = true;
  addMessage("user", message);
  input.value = "";
  addMessage("assistant", "思考中…（对话要过 LLM，最长可能到 60 秒）");

  try {
    const payload = { session_id: state.session, message };
    if (state.image) payload.image = state.image;
    const resp = await api("/api/chat", { method: "POST", body: payload });
    $("#chatLog").lastElementChild.remove();
    addMessage("assistant", resp.reply || "(空回复)", resp.tool_calls);
  } catch (err) {
    $("#chatLog").lastElementChild.remove();
    addMessage("err", err.message);
  } finally {
    state.busy = false;
    $("#btnSend").disabled = false;
  }
}

/* ---------------------------------------------------------------- 绑定 */

function bind() {
  $("#gwApply").addEventListener("click", applyTarget);
  $("#gwTarget").addEventListener("keydown", (e) => { if (e.key === "Enter") applyTarget(); });
  $("#btnHealth").addEventListener("click", refreshHealth);
  $("#btnRefreshTools").addEventListener("click", refreshTools);

  $("#fileInput").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try { await setImage(await compressToDataUrl(file), file.name); }
    catch (err) { $("#imgInfo").textContent = "读取失败：" + err.message; }
  });
  $("#btnSample").addEventListener("click", async () => {
    await setImage(drawSampleImage(), "示例图（合成，仅验证链路）");
  });
  $("#btnClearImg").addEventListener("click", clearImage);

  $("#toolSelect").addEventListener("change", renderParamForm);
  $("#btnInvoke").addEventListener("click", invokeSelected);
  $("#btnSelfCheck").addEventListener("click", selfCheck);

  $("#btnSend").addEventListener("click", sendChat);
  $("#chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendChat();
  });
  $("#btnClearChat").addEventListener("click", () => {
    state.session = newSessionId();
    localStorage.setItem("gwSession", state.session);
    $("#chatLog").innerHTML = "";
    addMessage("assistant", "已开启新会话（网关侧历史按 session_id 隔离）。");
  });

  window.addEventListener("resize", () => {
    if (state.lastBoxes && state.lastBoxes.length) drawBoxes(state.lastBoxes);
  });
}

(async function init() {
  bind();
  await loadConfig();
  if (state.target) $("#gwTarget").value = state.target;
  addMessage("assistant", "演示页就绪。先在上方选择一张图片，然后可以直接提问，或做工具直调。");
  await refreshHealth();
})();
