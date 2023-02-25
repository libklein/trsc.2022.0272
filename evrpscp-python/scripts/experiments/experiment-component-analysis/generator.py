# coding=utf-8
from pathlib import Path

from evrpscp import DiscretizedInstance
from evrpscp.models import Battery, Charger
from evrpscp.models.instance import plot_fleet_tours
from evrpscp.parsing.pelletier import parsePelletier

instance_base_path = Path("./data/pelletier/Short-routes/1_3v_Summer.dump.d")

inst = parsePelletier(Path(instance_base_path / "instance/1_3v.txt"),
                      Path(instance_base_path / "instance/Parameters/"))
instance = DiscretizedInstance.DiscretizeInstance(inst, period_duration=30.0)

print("--------------------------Tours-----------------------------------")
print(plot_fleet_tours(instance.tours))
print("---------------------------------------------------------------------")

import column_generation.cli

column_generation.cli.run_instance(instance, "")
