#!/bin/bash

# From https://github.com/bengreenier/docker-xvfb/

Xvfb :99 -ac -screen 0 "$XVFB_RES" -nolisten tcp $XVFB_ARGS &
XVFB_PROC=$!
sleep 1
export DISPLAY=:99
"$@"
kill $XVFB_PROC
