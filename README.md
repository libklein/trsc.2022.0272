# trsc.2022.0272 

## Structure

* `column_generation`: Implementation of the column generation algorithm (main solver). Requires `evrpscp-python`
  and `frvcp-cpp`.
* `dyn_tour_mip`: Implementation of the (compact) MIP formulation (See Appendix E
  in [the paper](https://arxiv.org/abs/2201.03972)). Requires `evrpscp-python`.
* `evrpscp`: Python library that defines models, provides instance parsers, and implements utility functions.
* `frvcp-cpp`: C++ implementation of the subproblem (labeling algorithm).
* `pelletier`: Implementation of the MIP proposed
  in [Pelletier et al. (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0191261517308871). Not directly
  related.
* `scripts`: Contains instance generation scripts. The parameters can be found in the paper.
* `examples`: Contains example instances, instance generation scripts, and run configurations.
* `data`: Contains data from [Pelletier et al. (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0191261517308871). Useful for testing.

## Requirements

* CPlex 22.1
* GCC >= 11
* CMake >= 3.15
* Boost >= 1.75
* PyBind11
* Python 3.8

## Installation

* Create Virtual Environment
* Install requirements
* Make sure to install CPlex properly:

```bash
cd /path/to/cplex
cd python
python setup.py install
```

* Run CMake/Build native library

```bash
cd frvcp-cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="RELEASE" -DPython_ROOT_DIR="<path/to/venv>"
cmake --build .
```

This will create a file called `evspnl.cpython-{python_version}-x86_64-linux-gnu.so` file in `./bindings`.
This is the native library.

## Running

Add the build path to the PYTHONPATH environment variable. Alternatively, just copy it to the directory from which
you will run your experiments.

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

Then proceed and add the main directory to your PYTHONPATH as well. This allows python to find the module when
importing.

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Usage

Detailed usage instructions can be found in the respective README files of each package. A summary of the options
available for each package can be obtained by running

```bash
python -m <package> --help
```

### Input

All provided solvers read instances via *solution dumps*. See the `examples/instances`
and `examples/instance-generation` directories for examples.

### Output

The solvers output a `JSON` file with the following structure:

```json

{
  "SolutionInfo": {
    "Runtime": "Runtime in seconds (float)",
    "ObjVal": "Best integer solution (float)",
    "ObjBound": "Tightest lower bound (float)",
    "RootLB": "Root node lower bound (float)",
    "MIPGap": "MIP gap (|bestbound-bestinteger|/(1e-10+|bestinteger|)) (float)",
    "IterCount": "Number of solved nodes (int)",
    "NodeCount": "Number of created nodes (int)"
  },
  "IterPerNode": "unused",
  "ScheduleDetails": {
    "Cost": "Total cost of the solution (float)",
    "TourCost": "Total tour cost of the solution (fix costs) (float)",
    "EnergyCost": "Total energy cost of the solution (float)",
    "DegradationCost": "Total degradation cost of the solution (float)",
    "TotalCharge": "Total amount of energy restored (float)"
  }
}
```
