#!/bin/bash

if [ $# -eq 0 ]; then
	unset LD_PRELOAD;
	python main.py
fi

if [ "$1" = "viewer" ]; then
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so python main.py --viewer
fi

if [ "$1" = "render" ]; then
	unset LD_PRELOAD;
	python main.py --render
fi

