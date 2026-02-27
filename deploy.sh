#!/bin/bash

# Server management script for API and LiveKit voice agent
# Usage: ./deploy.sh start|stop|restart [api|voice|all]
#   api   - FastAPI server only (port 8090)
#   voice - LiveKit voice agent only
#   all   - Both services (default)

API_PID_FILE="server.pid"
API_LOG_FILE="server.log"
VOICE_PID_FILE="voice.pid"
VOICE_LOG_FILE="voice.log"

# Activating venv
source .venv/bin/activate

start_api() {
    if [ -f "$API_PID_FILE" ] && kill -0 $(cat "$API_PID_FILE") 2>/dev/null; then
        echo "API server is already running (PID: $(cat "$API_PID_FILE"))"
        return 1
    fi
    echo "Starting API server..."
    nohup uvicorn app.main:app --host 0.0.0.0 --port 8090 > "$API_LOG_FILE" 2>&1 &
    echo $! > "$API_PID_FILE"
    echo "API server started (PID: $(cat "$API_PID_FILE"))"
}

start_voice() {
    if [ -f "$VOICE_PID_FILE" ] && kill -0 $(cat "$VOICE_PID_FILE") 2>/dev/null; then
        echo "Voice agent is already running (PID: $(cat "$VOICE_PID_FILE"))"
        return 1
    fi
    echo "Starting voice agent..."
    nohup python -m app.livekit_agent start > "$VOICE_LOG_FILE" 2>&1 &
    echo $! > "$VOICE_PID_FILE"
    echo "Voice agent started (PID: $(cat "$VOICE_PID_FILE"))"
}

stop_process() {
    local pid_file="$1"
    local label="$2"

    if [ ! -f "$pid_file" ]; then
        echo "$label is not running (no PID file found)"
        return 1
    fi
    PID=$(cat "$pid_file")
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "$label is not running (PID $PID not found)"
        rm -f "$pid_file"
        return 1
    fi
    echo "Stopping $label (PID: $PID)..."
    kill "$PID"
    # Wait for process to stop
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing $label..."
        kill -9 "$PID"
    fi
    rm -f "$pid_file"
    echo "$label stopped"
}

stop_api() {
    stop_process "$API_PID_FILE" "API server"
}

stop_voice() {
    stop_process "$VOICE_PID_FILE" "Voice agent"
}

SERVICE="${2:-all}"

case "$1" in
    start)
        case "$SERVICE" in
            api)   start_api ;;
            voice) start_voice ;;
            all)   start_api; start_voice ;;
            *)     echo "Unknown service: $SERVICE"; exit 1 ;;
        esac
        ;;
    stop)
        case "$SERVICE" in
            api)   stop_api ;;
            voice) stop_voice ;;
            all)   stop_api; stop_voice ;;
            *)     echo "Unknown service: $SERVICE"; exit 1 ;;
        esac
        ;;
    restart)
        case "$SERVICE" in
            api)   stop_api; sleep 2; start_api ;;
            voice) stop_voice; sleep 2; start_voice ;;
            all)   stop_api; stop_voice; sleep 2; start_api; start_voice ;;
            *)     echo "Unknown service: $SERVICE"; exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 {start|stop|restart} [api|voice|all]"
        exit 1
        ;;
esac