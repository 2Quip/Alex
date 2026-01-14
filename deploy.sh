#!/bin/bash

# Server management script for app/main.py
# Usage: ./server_manager.sh start|stop|restart

PID_FILE="server.pid"
LOG_FILE="server.log"

# Activating venv
source .venv/bin/activate

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Server is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    echo "Starting server..."
    nohup uvicorn app.main:app --host 0.0.0.0 --port 8090 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Server started (PID: $(cat "$PID_FILE"))"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Server is not running (no PID file found)"
        return 1
    fi
    PID=$(cat "$PID_FILE")
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Server is not running (PID $PID not found)"
        rm -f "$PID_FILE"
        return 1
    fi
    echo "Stopping server (PID: $PID)..."
    kill "$PID"
    # Wait for process to stop
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing server..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "Server stopped"
}

restart() {
    stop
    sleep 2
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac