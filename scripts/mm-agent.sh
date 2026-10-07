#!/usr/bin/env bash
#
# mm-agent 一键启动 / 停止 / 查看状态。
#
#   ./scripts/mm-agent.sh start              # 起四个节点（网关 + kb + 两个视觉节点）
#   ./scripts/mm-agent.sh start --mock       # mock 模式：不需要任何模型权重，秒起，
#                                            #   适合只调前端的人
#   ./scripts/mm-agent.sh start --with-vllm  # 连 vLLM 一起起（需要权重 + 显卡）
#   ./scripts/mm-agent.sh start --only kb    # 只起指定节点：kb / gateway / vision-fast /
#                                            #   vision-heavy / vllm，逗号分隔任意组合
#   ./scripts/mm-agent.sh stop
#   ./scripts/mm-agent.sh status
#   ./scripts/mm-agent.sh restart
#
# 配置来源（优先级：显式环境变量 > 仓库根目录 .env > 下方默认值）：
#   .env        统一入口，模板见 .env.example。端口 / 节点 IP / vLLM 参数 /
#               权重路径 / VLLM_VENV 都在这儿改，不必每次在命令行前导。
#   命令行前导   临时覆盖单个值，例：VLLM_VENV=/path ./scripts/mm-agent.sh start
#
# 支持的变量（都有默认值，见下方赋值处）：
#   MM_PY         仓库 venv 的 python，默认 <repo>/.venv/bin/python
#   MM_RUN_DIR    pid / 日志目录，默认 <repo>/.run
#   VLLM_VENV     vLLM 所在 venv，默认 ~/vllm-venv
#   *_PORT        各节点端口
#   VLLM_*        vLLM 的模型路径 / 服务名 / 显存参数

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 加载仓库根目录的 .env --------------------------------------------------
# 让端口 / VLLM_VENV / 显存参数这些和节点配置用同一份文件，不必每次命令行前导。
# 优先级与 common/config.py 保持一致：**已存在的环境变量一律不覆盖**，
# 所以 `VLLM_VENV=/x ./scripts/mm-agent.sh start` 这种临时覆盖仍然生效。
# 不用 `source` 是刻意的：.env 里写错了不该把整个脚本带崩，且 source 会覆盖
# 已有变量，与上面那条优先级相反。
load_env_file() {
  local f="$ROOT/.env" raw key val
  [[ -f "$f" ]] || return 0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    raw="${raw%$'\r'}"                            # 容忍 CRLF
    raw="${raw#"${raw%%[![:space:]]*}"}"          # 去行首空白
    raw="${raw#"export "}"                        # 容忍 export KEY=...
    if [[ "$raw" != *=* || "$raw" == '#'* ]]; then
      continue                                    # 跳过空行与整行注释
    fi
    key="${raw%%=*}"; val="${raw#*=}"
    key="${key//[[:space:]]/}"                    # 键里不该有空白，去掉
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$val" == '"'*'"' || "$val" == "'"*"'" ]]; then
      val="${val:1:${#val}-2}"                    # 成对引号：引号内原样保留
    else
      val="${val%%[[:space:]]#*}"                 # 砍掉行尾 " # 注释"
      val="${val%"${val##*[![:space:]]}"}"        # 去行尾空白
    fi
    if [[ -z "${!key:-}" ]]; then                 # 显式环境变量优先，不覆盖
      export "$key=$val"
    fi
  done < "$f"
}
load_env_file

RUN_DIR="${MM_RUN_DIR:-$ROOT/.run}"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"

PY="${MM_PY:-$ROOT/.venv/bin/python}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"

GATEWAY_PORT="${GATEWAY_PORT:-8000}"
VISION_FAST_PORT="${VISION_FAST_PORT:-8101}"
VISION_HEAVY_PORT="${VISION_HEAVY_PORT:-8102}"
KB_PORT="${KB_PORT:-8103}"
VLLM_PORT="${VLLM_PORT:-8001}"

# 单机四进程共用一张卡，默认压到 0.5；只有 vLLM 独占整卡时才可以调高
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-}"          # 空 = 用下面的默认；见参数解析后的解析逻辑
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-}"          # 空 = 按模型目录名推导
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"    # 空 = 优先用模型目录自带的那份

export GATEWAY_PORT VISION_FAST_PORT VISION_HEAVY_PORT KB_PORT VLLM_PORT VLLM_MODEL_NAME

MOCK=0
WITH_VLLM=0
ONLY=""
CMD=""
MODEL_ARG=""
while (( $# )); do
  case "$1" in
    --mock)      MOCK=1 ;;
    --with-vllm) WITH_VLLM=1 ;;
    --only)      shift; ONLY="${1:-}" ;;
    --only=*)    ONLY="${1#--only=}" ;;
    --model)     shift; MODEL_ARG="${1:-}" ;;
    --model=*)   MODEL_ARG="${1#--model=}" ;;
    start|stop|restart|status) CMD="$1" ;;
    -h|--help)   CMD="help" ;;
    *) echo "未知参数：$1（用 -h 看帮助）" >&2; exit 1 ;;
  esac
  shift
done
CMD="${CMD:-start}"
ONLY="${ONLY:-}"

# ---- 模型相关参数解析 -------------------------------------------------------
# --model 优先于 VLLM_MODEL_PATH；都没给就退回文档里的 Qwen2.5 默认路径。
[[ -n "$MODEL_ARG" ]] && VLLM_MODEL_PATH="$MODEL_ARG"
if [[ -z "$VLLM_MODEL_PATH" ]]; then
  VLLM_MODEL_PATH="$ROOT/weights/Qwen2.5-VL-3B-Instruct-AWQ"
elif [[ "$VLLM_MODEL_PATH" != /* ]]; then
  VLLM_MODEL_PATH="$ROOT/$VLLM_MODEL_PATH"    # 允许传相对仓库根目录的路径
fi

# 服务名：没显式给就按目录名推。关键不是取什么名，而是 vLLM 的
# --served-model-name 和网关的 VLLM_MODEL_NAME **必须是同一个**，否则模型调不动。
[[ -z "$VLLM_MODEL_NAME" ]] && \
  VLLM_MODEL_NAME="$(basename "$VLLM_MODEL_PATH" | tr '[:upper:]' '[:lower:]')"

# chat template：**优先用模型目录自带的那份**。Qwen3-VL 系列的权重不带内嵌模板，
# 必须外指；而仓库里那份 qwen25_tools_chat_template.jinja 只适用于 Qwen2.5，
# 拿它套 Qwen3-VL 会让模型看不到工具定义。所以按目录里有没有来决定。
if [[ -z "$VLLM_CHAT_TEMPLATE" ]]; then
  if [[ -f "$VLLM_MODEL_PATH/chat_template.jinja" ]]; then
    VLLM_CHAT_TEMPLATE="$VLLM_MODEL_PATH/chat_template.jinja"
  else
    VLLM_CHAT_TEMPLATE="$ROOT/llm_node/qwen25_tools_chat_template.jinja"
  fi
fi

usage() {
  cat <<'EOF'
mm-agent 一键启动 / 停止 / 查看状态。

  ./scripts/mm-agent.sh start              # 起四个节点（网关 + kb + 两个视觉节点）
  ./scripts/mm-agent.sh start --mock       # mock 模式：不需要任何模型权重，秒起，
                                           #   适合只调前端的人
  ./scripts/mm-agent.sh start --with-vllm  # 连 vLLM 一起起（需要权重 + 显卡）
  ./scripts/mm-agent.sh start --only kb    # 只起指定节点，逗号分隔任意组合：
                                           #   kb / gateway / vision-fast / vision-heavy / vllm
  ./scripts/mm-agent.sh stop
  ./scripts/mm-agent.sh restart
  ./scripts/mm-agent.sh status

  --model DIR   指定 vLLM 加载哪个权重目录（相对仓库根目录或绝对路径）。
                服务名与 chat template 会**自动**从它推导：
                  · 服务名 = 目录名小写，同时喂给网关，两边必然一致
                  · template 优先用该目录自带的 chat_template.jinja
                    （Qwen3-VL 必须用自带的；仓库里那份是 Qwen2.5 专用）
                不传则用 VLLM_MODEL_PATH，再没有就退回 weights 里的 Qwen2.5 默认

配置来源（优先级：显式环境变量 > 仓库根目录 .env > 下面的默认值）：
  **日常改仓库根目录的 .env 就够了**，模板见 .env.example，下面这些都能在里面设；
  命令行前导只用于临时覆盖某个值。

  脚本自己的：
  MM_PY         仓库 venv 的 python，默认 <repo>/.venv/bin/python
  MM_RUN_DIR    pid / 日志目录，默认 <repo>/.run
  VLLM_VENV     vLLM 所在 venv，默认 ~/vllm-venv
  GATEWAY_PORT / VISION_FAST_PORT / VISION_HEAVY_PORT / KB_PORT / VLLM_PORT
  节点与模型（由各节点自己读）：
  VLLM_MODEL_PATH / VLLM_MODEL_NAME / VLLM_CHAT_TEMPLATE
  VLLM_GPU_MEMORY_UTILIZATION / VLLM_MAX_MODEL_LEN
EOF
}

log()  { printf '\033[36m[mm-agent]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[mm-agent]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[mm-agent]\033[0m %s\n' "$*" >&2; exit 1; }

# 依赖检查
[[ -x "$PY" ]] || die "找不到仓库 venv 的 python：$PY
先建环境：uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
（网关和视觉节点共用这个 venv；vLLM 是另一个 venv，别混）"

# 可选依赖检查。只在真要起服务时做，stop / status 不啰嗦。
# PyMuPDF 只影响 read_document 读 PDF：缺了不挡节点启动，但很容易被无声忽略
# （前端传了 PDF 才发现报错），所以主动提一句。
if [[ "$CMD" == "start" || "$CMD" == "restart" ]]; then
  "$PY" -c 'import pymupdf' >/dev/null 2>&1 || warn "没装 pymupdf —— vision-heavy 的 read_document 读不了 PDF（文本 / 图片仍可用）
  补装：uv pip install --python .venv/bin/python -r requirements.txt"
fi

curl_ok() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }

# ---- 要起哪些节点 -----------------------------------------------------------
# 默认全起（四个节点；vLLM 仅 --with-vllm 时）。--only 逗号分隔，精确到节点，
# 例：--only kb   --only gateway   --only kb,gateway   --only vision-fast
declare -a WANT_NODES=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra WANT_NODES <<< "$ONLY"
  for n in "${WANT_NODES[@]}"; do
    case "$n" in
      kb|gateway|vision-fast|vision-heavy|vllm) ;;
      *) die "未知节点名：$n（可选：kb / gateway / vision-fast / vision-heavy / vllm）" ;;
    esac
  done
else
  WANT_NODES=("vision-fast" "vision-heavy" "kb" "gateway")
  (( WITH_VLLM )) && WANT_NODES+=("vllm")
fi
has_node() { local n; for n in "${WANT_NODES[@]}"; do [[ "$n" == "$1" ]] && return 0; done; return 1; }

# 源码 / 配置是不是比进程新。
#
# 这是本地开发最容易踩的坑：改了代码（或 .env）再敲 start，**已在运行的节点会被
# 直接跳过**，于是新工具、新行为根本没生效，而界面上什么都看不出来——只会觉得
# "怎么还是老样子"。所以发现这种情况必须明确喊出来。
newer_than_proc() {  # $1=pid，其余 = 文件或目录（目录按 *.py 递归比 mtime）
  local pid="$1"; shift
  local started epoch
  started="$(ps -o lstart= -p "$pid" 2>/dev/null)" || return 1
  [[ -n "$started" ]] || return 1
  epoch="$(date -d "$started" +%s 2>/dev/null)" || return 1
  local t m
  for t in "$@"; do
    if [[ -d "$t" ]]; then
      if find "$t" -name '*.py' -newermt "@$epoch" -print -quit 2>/dev/null | grep -q .; then
        return 0
      fi
    elif [[ -f "$t" ]]; then
      m="$(stat -c %Y "$t" 2>/dev/null)" || continue
      [[ -n "$m" && "$m" -gt "$epoch" ]] && return 0
    fi
  done
  return 1
}


# 起一个 uvicorn 进程。$1=名字 $2=端口 $3=应用 $4=mock(0/1)
#
# 注意这里不套子 shell：套了的话 `$!` 拿到的是那层 shell 的 pid 而不是 uvicorn 的，
# 写进 pid 文件就是死的、stop 会失效。stdin 也接到 /dev/null，否则后台进程会一直
# 攥着父进程的 stdout（脚本被 `| tail` 之类管道包住时，管道永远不 EOF、挂住）。
spawn() {
  local name="$1" port="$2" app="$3" mock="$4"
  local pf="$PID_DIR/$name.pid"
  mkdir -p "$LOG_DIR" "$PID_DIR"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    local pid; pid="$(cat "$pf")"
    warn "$name 已在运行（pid $pid），跳过"
    if newer_than_proc "$pid" "$ROOT/${app%%.*}" "$ROOT/common"; then
      warn "  ↑ 但源码比这个进程新，改动没生效——先 stop 再 start（或 restart）"
    fi
    return 0
  fi
  log "启动 $name → :$port  (日志 $LOG_DIR/$name.log)"
  local oldpwd="$PWD"
  cd "$ROOT"
  NODE_MOCK="$mock" VLLM_MODEL_NAME="$VLLM_MODEL_NAME" VLLM_PORT="$VLLM_PORT" \
    nohup "$PY" -m uvicorn "$app" --host 0.0.0.0 --port "$port" \
    >"$LOG_DIR/$name.log" 2>&1 </dev/null &
  echo $! >"$pf"
  cd "$oldpwd"
}

# 等健康检查通过；$1=端口 $2=路径 $3=超时秒
wait_health() {
  local port="$1" path="$2" timeout="$3" t=0
  while (( t < timeout )); do
    curl_ok "http://127.0.0.1:$port$path" && return 0
    sleep 0.5; t=$((t + 1))
  done
  return 1
}

start_vllm() {
  local bin="$VLLM_VENV/bin/vllm"
  [[ -x "$bin" ]] || die "找不到 vLLM：$bin
用 VLLM_VENV 指定装了 vllm 的那个 venv（它和仓库 .venv 依赖冲突，装不到一起）。
不知道在哪的话，先搜一下：
  find \$HOME -maxdepth 5 -type f -name vllm -path '*/bin/*' 2>/dev/null"
  [[ -d "$VLLM_MODEL_PATH" ]] || die "找不到权重目录：$VLLM_MODEL_PATH
设置 VLLM_MODEL_PATH，或先用 --mock 起（mock 不需要权重）"
  mkdir -p "$LOG_DIR" "$PID_DIR"
  if curl_ok "http://127.0.0.1:$VLLM_PORT/v1/models"; then
    warn "vLLM 已在运行，跳过"
    local vpid; vpid="$(cat "$PID_DIR/vllm.pid" 2>/dev/null || true)"
    if [[ -n "$vpid" ]] && newer_than_proc "$vpid" "$ROOT/.env"; then
      warn "  ↑ 但 .env 比这个 vLLM 进程新：改的窗口 / 显存参数没生效，"
      warn "    模型仍是启动时那份（用它 --model / VLLM_* 启动时读到的值）"
    fi
    return 0
  fi
  log "启动 vLLM → :$VLLM_PORT"
  log "  权重    $VLLM_MODEL_PATH"
  log "  服务名  $VLLM_MODEL_NAME   （网关用同一个，两边必须一致）"
  log "  模板    $VLLM_CHAT_TEMPLATE"
  log "  显存    gpu-memory-utilization=$VLLM_GPU_MEMORY_UTILIZATION  max-model-len=$VLLM_MAX_MODEL_LEN"
  log "  首次启动要为 sm_120 现场编译 kernel，可能要几分钟。日志：$LOG_DIR/vllm.log"
  [[ -f "$VLLM_CHAT_TEMPLATE" ]] || die "chat template 不存在：$VLLM_CHAT_TEMPLATE"
  local oldpwd="$PWD"
  cd "$ROOT"
  VLLM_WSL2_ENABLE_PIN_MEMORY=1 nohup "$bin" serve "$VLLM_MODEL_PATH" \
    --served-model-name "$VLLM_MODEL_NAME" \
    --chat-template "$VLLM_CHAT_TEMPLATE" \
    --port "$VLLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    >"$LOG_DIR/vllm.log" 2>&1 </dev/null &
  echo $! >"$PID_DIR/vllm.pid"
  cd "$oldpwd"
  wait_health "$VLLM_PORT" "/v1/models" 600 \
    || die "vLLM 6 分钟没起来，看 $LOG_DIR/vllm.log"
}

do_start() {
  local mock=0
  (( MOCK )) && mock=1

  if (( WITH_VLLM )) && (( MOCK )); then
    warn "--with-vllm 和 --mock 互斥（mock 就是不要模型），vLLM 不起"
    WITH_VLLM=0
  fi

  if (( MOCK )); then
    log "MOCK 模式：返回占位结果，不加载任何模型（不用权重、不占显存）"
  elif has_node vllm; then
    start_vllm
  elif has_node gateway; then
    curl_ok "http://127.0.0.1:$VLLM_PORT/v1/models" \
      || warn "本机 :$VLLM_PORT 没探到 vLLM——若 vLLM 在远端属正常（记得 LLM_HOST 指过去），
       否则 /api/chat 会报「LLM 服务不可达」；只想看界面就加 --mock"
  fi

  if has_node vision-fast; then
    spawn vision-fast  "$VISION_FAST_PORT"  "vision_fast.server:app"   "$mock"
    wait_health "$VISION_FAST_PORT"  "/health" 120 || warn "vision-fast 健康检查超时"
  fi
  if has_node vision-heavy; then
    spawn vision-heavy "$VISION_HEAVY_PORT" "vision_heavy.server:app"  "$mock"
    wait_health "$VISION_HEAVY_PORT" "/health" 120 || warn "vision-heavy 健康检查超时"
  fi
  if has_node kb; then
    spawn kb-node "$KB_PORT" "kb_node.server:app" "$mock"
    wait_health "$KB_PORT" "/health" 60 || warn "kb-node 健康检查超时"
  fi
  if has_node gateway; then
    # 顺序有讲究：网关启动时会去发现各工具节点的工具，放在最后起
    spawn gateway "$GATEWAY_PORT" "llm_node.gateway:app" "$mock"
    wait_health "$GATEWAY_PORT" "/api/health" 60 || warn "网关健康检查超时"
  fi

  echo
  log "启动完成。健康状态："
  MM_MOCK="$mock" curl -s "http://127.0.0.1:$GATEWAY_PORT/api/health" \
    | MM_MOCK="$mock" "$PY" -c 'import sys,json,os
d=json.load(sys.stdin)
mock = os.environ.get("MM_MOCK") == "1"
print("  网关     ", d.get("gateway",{}).get("status"), d.get("gateway",{}).get("tools"))
for n,i in (d.get("nodes") or {}).items():
    print(f"  {n:<9}", "ok" if i.get("ok") else "FAIL", i.get("tools"), i.get("error") or "")
v=d.get("vllm") or {}
if mock:
    print("  vllm      未启动（mock 模式不调用模型，这是正常的）")
else:
    print("  vllm     ", "ok" if v.get("ok") else "FAIL", v.get("models") or v.get("error") or "")
print("  all_ok   ", d.get("all_ok"), "（mock 模式下 vllm 未起，all_ok=false 属正常）" if mock else "")' \
    2>/dev/null || warn "取健康状态失败，看 $LOG_DIR/gateway.log"

  echo
  log "接口："
  echo "  GET  http://127.0.0.1:$GATEWAY_PORT/api/health   存活与工具发现"
  echo "  GET  http://127.0.0.1:$GATEWAY_PORT/api/tools    工具清单"
  echo "  POST http://127.0.0.1:$GATEWAY_PORT/api/chat     对话（带 stream:true 走 SSE）"
  echo
  log "停止：./scripts/mm-agent.sh stop     日志：$LOG_DIR"
}

do_stop() {
  local stopped=0
  for name in gateway kb-node vision-heavy vision-fast vllm; do
    local f="$PID_DIR/$name.pid"
    [[ -f "$f" ]] || continue
    local pid; pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      log "停止 $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$pid" 2>/dev/null && { warn "$name 没退出，强杀"; kill -9 "$pid" 2>/dev/null || true; }
      stopped=$((stopped + 1))
    fi
    rm -f "$f"
  done
  (( stopped )) && log "已停止 $stopped 个进程" || log "没有在跑的进程"
}

do_status() {
  check() { # $1=名字 $2=url
    if curl_ok "$2"; then printf '  %-14s \033[32mup\033[0m   %s\n' "$1" "$2"
    else                    printf '  %-14s \033[31mdown\033[0m %s\n' "$1" "$2"; fi
  }
  log "服务状态："
  check "vllm"          "http://127.0.0.1:$VLLM_PORT/v1/models"
  check "vision-fast"   "http://127.0.0.1:$VISION_FAST_PORT/health"
  check "vision-heavy"  "http://127.0.0.1:$VISION_HEAVY_PORT/health"
  check "kb-node"       "http://127.0.0.1:$KB_PORT/health"
  check "gateway"       "http://127.0.0.1:$GATEWAY_PORT/api/health"
  if curl_ok "http://127.0.0.1:$GATEWAY_PORT/api/health"; then
    echo
    curl -s "http://127.0.0.1:$GATEWAY_PORT/api/health" | "$PY" -m json.tool 2>/dev/null | head -40
  fi
}

case "$CMD" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; echo; do_start ;;
  status)  do_status ;;
  help)    usage ;;
esac
