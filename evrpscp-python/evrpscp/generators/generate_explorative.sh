#!/usr/bin/env bash

NUM_INSTANCES=50
# Fix:
NUM_VEHICLES=4
NUM_CHARGERS=1
CHARGER_CAPACITY=1
FULL_CHARGE_DUR=((4 * 60)) # 4 hours
PAUSE_BEFORE_TOUR=8 # 4 hours


# Variable
TOUR_LENGTH=(4 8 12) # Between 0.5 and 1.5 * 4 hours
MIN_TW_LEN=(4 )
MAX_TW_LEN=()
MIN_CONSUMPTION=
MAX_CONSUMPTION=
NUM_TOURS=

python ./explorative_instances.py -o 'instances' --suffix='v1' --num-instances=${NUM_INSTANCES}
--num-tours=${num_tours} --num-chargers=${NUM_CHARGERS} --num-vehicles=${NUM_VEHICLES}
--min-soc-consumption=${min_consumption} --max-soc-consumption=${max_consumption}
