#!/usr/bin/env bash
set -e


# Wait for redis
#until redis-cli -h redis ping | grep PONG; do
#echo "Waiting for redis..."
#sleep 1
#done


# Launch app
python src/app.py