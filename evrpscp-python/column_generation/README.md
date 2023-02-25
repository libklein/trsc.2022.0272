# column_generation

Main solver for the EVRPSCP problem. Implements the column generation algorithm detailed in
the [paper](https://arxiv.org/abs/2201.03972).

## Usage

The solver can be run from the command line with the following command:

```bash
python column_generation.py --instance <instance_path> --output <output_path> --time-limit <time_limit> --seed <seed>
```

For more information on the parameters, run:

```bash
python column_generation.py --help
```
