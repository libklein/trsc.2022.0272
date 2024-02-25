#!/usr/bin/env bash

if [ $# -ne 1 ]; then
	echo "Usage: $0 <path-to-instance-dump-directory>"
	exit 1
fi

instance_path="$1"

python -m column_generation --time-limit 300 -o solutions -d ${instance_path} -l logging.yaml  --print-solution
