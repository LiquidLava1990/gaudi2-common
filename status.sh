#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/common.env"
echo "=== active model container ==="; docker ps --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Command}}' 2>/dev/null
echo "=== endpoint ==="; echo "http://$BIND_IP:$PORT/v1  (api-key in common.env)"
echo "=== available model configs ==="; ls "$(dirname "$(readlink -f "$0")")/models" | sed 's/.env$//'
