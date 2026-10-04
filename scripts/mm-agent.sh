#!/usr/bin/env bash
#
# mm-agent 一键启动 / 停止 / 查看状态。
#
#   ./scripts/mm-agent.sh start              # 起三个节点（网关 + 两个视觉节点）
#   ./scripts/mm-agent.sh start --mock       # mock 模式：不需要任何模型权重，秒起，
#                                            #   适合只调前端的人
#   ./scripts/mm-agent.sh start --with-vllm  # 连 vLLM 一起起（需要权重 + 显卡）
#   ./scripts/mm-agent.sh stop
#   ./scripts/mm-agent.sh status
#   ./scripts/mm-agent.sh restart
#
# 环境变量（都有默认值，见下方赋值处）：
#   MM_PY         仓库 venv 的 python，默认 <repo>/.venv/bin/python
#   VLLM_VENV     vLLM 所在 venv，默认 ~/vllm-venv
#   *_PORT        各节点端口
#   VLLM_*        vLLM 的模型路径 / 服务名 / 显存参数

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${MM_RUN_DIR:-$ROOT/.run}"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"

PY="${MM_PY:-$ROOT/.venv/bin/python}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"

GATEWAY_PORT="${GATEWAY_PORT:-8000}"
VISION_FAST_PORT="${VISION_FAST_PORT:-8101}"
VISION_HEAVY_PORT="${VISION_HEAVY_PORT:-8102}"
VLLM_PORT="${VLLM_PORT:-8001}"

# 单机四进程共用一张卡，默认压到 0.5；只有 vLLM 独占整卡时才可以调高
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-}"          # 空 = 用下面的默认；见参数解析后的解析逻辑
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-}"          # 空 = 按模型目录名推导
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"    # 空 = 优先用模型目录自带的那份

export GATEWAY_PORT VISION_FAST_PORT VISION_HEAVY_PORT VLLM_PORT VLLM_MODEL_NAME

MOCK=0
WITH_VLLM=0
CMD=""
MODEL_ARG=""
while (( $# )); do
  case "$1" in
    --mock)      MOCK=1 ;;
    --with-vllm) WITH_VLLM=1 ;;
    --model)     shift; MODEL_ARG="${1:-}" ;;
    --model=*)   MODEL_ARG="${1#--model=}" ;;
    start|stop|restart|status) CMD="$1" ;;
    -h|--help)   CMD="help" ;;
    *) echo "未知参数：$1（用 -h 看帮助）" >&2; exit 1 ;;
  esac
  shift
done
CMD="${CMD:-start}"

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

  ./scripts/mm-agent.sh start              # 起三个节点（网关 + 两个视觉节点）
  ./scripts/mm-agent.sh start --mock       # mock 模式：不需要任何模型权重，秒起，
                                           #   适合只调前端的人
  ./scripts/mm-agent.sh start --with-vllm  # 连 vLLM 一起起（需要权重 + 显卡）
  ./scripts/mm-agent.sh start --with-vllm --model weights/Qwen3-VL-4B-Instruct-AWQ-4bit
  ./scripts/mm-agent.sh stop
  ./scripts/mm-agent.sh restart
  ./scripts/mm-agent.sh status

  --model DIR   指定 vLLM 加载哪个权重目录（相对仓库根目录或绝对路径）。
                服务名与 chat template 会**自动**从它推导：
                  · 服务名 = 目录名小写，同时喂给网关，两边必然一致
                  · template 优先用该目录自带的 chat_template.jinja
                    （Qwen3-VL 必须用自带的；仓库里那份是 Qwen2.5 专用）
                不传则用 VLLM_MODEL_PATH，再没有就退回 weights 里的 Qwen2.5 默认

环境变量（都有默认值）：
  MM_PY         仓库 venv 的 python，默认 <repo>/.venv/bin/python
  VLLM_VENV     vLLM 所在 venv，默认 ~/vllm-venv
  GATEWAY_PORT / VISION_FAST_PORT / VISION_HEAVY_PORT / VLLM_PORT
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

curl_ok() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }

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
    warn "$name 已在运行（pid $(cat "$pf")），跳过"
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
设置 VLLM_VENV 指向装了 vllm==0.26.0 的 venv（它和仓库 .venv 装不到一起）"
  [[ -d "$VLLM_MODEL_PATH" ]] || die "找不到权重目录：$VLLM_MODEL_PATH
设置 VLLM_MODEL_PATH，或先用 --mock 起（mock 不需要权重）"
  mkdir -p "$LOG_DIR" "$PID_DIR"
  if curl_ok "http://127.0.0.1:$VLLM_PORT/v1/models"; then
    warn "vLLM 已在运行，跳过"; return 0
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
    warn "--with-vllm 和 --mock 互斥（mock 就是不要模型），只按 --mock 起"
    WITH_VLLM=0
  fi

  if (( MOCK )); then
    log "MOCK 模式：返回占位结果，不加载任何模型（不用权重、不占显存）"
  elif (( WITH_VLLM )); then
    start_vllm
  else
    curl_ok "http://127.0.0.1:$VLLM_PORT/v1/models" \
      || warn "vLLM(:$VLLM_PORT) 没起，/api/chat 会报「LLM 服务不可达」。
       要么加 --with-vllm，要么先自己起 vLLM；只想看界面就加 --mock"
  fi

  # 顺序有讲究：网关启动时会去发现两个视觉节点的工具
  spawn vision-fast  "$VISION_FAST_PORT"  "vision_fast.server:app"   "$mock"
  spawn vision-heavy "$VISION_HEAVY_PORT" "vision_heavy.server:app"  "$mock"
  wait_health "$VISION_FAST_PORT"  "/health" 120 || warn "vision-fast 健康检查超时"
  wait_health "$VISION_HEAVY_PORT" "/health" 120 || warn "vision-heavy 健康检查超时"

  spawn gateway "$GATEWAY_PORT" "llm_node.gateway:app" "$mock"
  wait_health "$GATEWAY_PORT" "/api/health" 60 || warn "网关健康检查超时"

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
  for name in gateway vision-heavy vision-fast vllm; do
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
