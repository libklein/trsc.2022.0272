#!/usr/bin/env bash

if [ $# -ne 1 ]; then
	echo "Usage: $0 <path-to-instance-dump>"
	exit 1
fi

instance_name="$1"

python -m dyn_tour_mip -l logs -o solutions -d "${instance_name}" --time-limit 300
