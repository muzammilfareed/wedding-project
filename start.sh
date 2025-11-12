#!/bin/bash

# Kill any process using port 8888
PORT=8888
PID=$(lsof -t -i:$PORT)
if [ ! -z "$PID" ]; then
    echo "Killing process on port $PORT (PID $PID)"
    kill -9 $PID
fi

# Start FastAPI with Gunicorn in the background and auto-restart
while true; do
    echo "Starting FastAPI server..."
    nohup gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT > fastapi.log 2>&1 &
    GUN_PID=$!
    echo "Gunicorn started with PID $GUN_PID"
    wait $GUN_PID
    echo "Server crashed. Restarting in 5 seconds..."
    sleep 5
done
