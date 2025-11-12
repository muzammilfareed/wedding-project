#!/bin/bash
while true; do
  gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8888
  echo "Server crashed. Restarting in 5 seconds..."
  sleep 5
done
