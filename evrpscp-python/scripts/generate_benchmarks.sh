#!/bin/zsh
# Generate benchmark instances
# Between 1 and 2 chargers

# Balanced instance
# On average as much time as nessesary before each tour

OUTPUT_DIRECTORY="output"
if ! [ -e "${OUTPUT_DIRECTORY}" ]; then
    mkdir -p ${OUTPUT_DIRECTORY}
fi

for num_veh in $(seq 3 9); do
    for num_tours in $(seq 2 5); do
        python ./evrpscp-py/evrpscp/generators/benchmark_instances.py -o ${OUTPUT_DIRECTORY} \
        --num-instances 10 \
        --num-vehicles ${num_veh} \
        --num-tours ${num_tours} \
        --num-chargers 2 \
        --charger-capacity 1 3 \
        --full-charge-duration 180 660 \
        --tour-discharge 0.3 0.6 \
        --tour-cost 10 50 \
        --free-time-before-tour 300 450 \
        --tour-duration 240 330 \
        --energy-price 0.1 0.7 \
        --check-feasibility \
        --seed "$(date +%d-%m)" \
        --suffix "balanced"
    done
done
