#!/usr/bin/env python
# coding=utf-8
import random
from copy import copy, deepcopy
from typing import List, Optional, Tuple, Dict

import pandas as pd
#import pandasgui as pdgui
import seaborn as sns
from matplotlib import pyplot as plt
import re
from json import load as parse_json
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
from evrpscp.generators.tiny_instances import *
import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.models import *
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper, create_result_df


@click.group()
def cli():
    pass

@cli.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--suffix', default='')
@click.option('--num-instances', default=500, type=int)
@click.option('--num-tours', default=list([2]), type=int, multiple=True)
@click.option('--num-chargers', default=list([1]), type=int, multiple=True)
@click.option('--num-vehicles', default=list([1]), type=int, multiple=True)
@click.option('--energy-price-variance', type=float, default=list([0.1]), multiple=True)
@click.option('--min-soc-consumption', type=float, default=0.25)
@click.option('--max-soc-consumption', type=float, default=0.75)
@click.option('--min-charger-capacity', type=int, default=1)
@click.option('--max-charger-capacity', type=int, default=1)
@click.option('--seed')
def generate_instances(output_directory: Path, suffix: str, num_tours: List[int], num_instances: int, seed: Optional[str],
                 energy_price_variance: List[float],
                 num_chargers: List[int], num_vehicles: List[int],
                 min_charger_capacity: int, max_charger_capacity: int, min_soc_consumption: float,
                 max_soc_consumption: float):
    """
    Generate several instances with varying energy price variance.
    """
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    if any(c not in (1, 2) for c in num_chargers):
        raise NotImplementedError('Only 1/2 chargers are supported at the moment')

    seed_generator = random.Random(seed)
    mean_ep = (energy_price_variance[-1] - energy_price_variance[0]) / 2 if len(energy_price_variance) > 0 else 1.0

    for instance_id in range(num_instances):
        for veh_count in num_vehicles:
            for tour_count in num_tours:
                for charger_count in num_chargers:
                    instance = generate_small_instance(num_tours=tour_count, seed=str(seed_generator.random()),
                                                       num_chargers=charger_count,
                                                       energy_price=Parameter(0.0001, 1.0),
                                                       charger_capacity=Parameter(min_charger_capacity, max_charger_capacity),
                                                       consumption=Parameter(min_soc_consumption, max_soc_consumption),
                                                       num_vehicles=veh_count)

                    # Instance carries energy price distribution
                    for energy_price_var in energy_price_variance:
                        min_ep, max_ep = (mean_ep - (energy_price_var/2), mean_ep + (energy_price_var/2))
                        instance_copy = deepcopy(instance)

                        for p in instance_copy.periods:
                            p.energyPrice = min_ep + (max_ep - min_ep) * p.energyPrice
                            delattr(p, 'pred')
                            delattr(p, 'succ')

                        # Write to file
                        name = f'tiny_{instance_copy.param.fleetSize}v_{tour_count}t_{charger_count}c_' \
                               f'{energy_price_var:.2}epv_run-{instance_id}_{suffix}'
                        Dump.DumpSchedulingInstance(directory=Path(str(output_directory / name) + '.dump.d'),
                                                    instance_name=name, instance=instance_copy, is_discretized=True)

@cli.command()
@click.argument('solutionl', type=click.Path(exists=True, dir_okay=True, file_okay=False), nargs=1)
@click.argument('solutionr', type=click.Path(exists=True, dir_okay=True, file_okay=False), nargs=1)
def compare_results(solutionl: Path, solutionr: Path):
    solutionl, solutionr = Path(solutionl), Path(solutionr)
    def process_sols(solutions: List[Path]) -> pd.DataFrame:
        results_df = create_result_df(solutions, re.compile(
            "tiny_(?P<fleet_size>\d+)v_(?P<tours>\d+)t_(?P<num_chargers>\d+)c_(?P<energy_price_variance>\d+\.\d+)epv_run-(?P<run>\d+)_(?P<suffix>.*)"))
        results_df = results_df[~results_df['infeasible'] & ~results_df['unsolved']]
        return results_df

    solutions_lhs = [x for x in Path(solutionl).rglob('*_branching-bench*') if x.is_dir()]
    solutions_rhs = [x for x in Path(solutionr).rglob('*_branching-bench*') if x.is_dir()]
    results_l_df = process_sols(solutions_lhs)
    results_r_df = process_sols(solutions_rhs)
    results_l_df['version'] = solutionl.name if solutionl.name != 'results' else solutionl.parent.name
    results_r_df['version'] = solutionr.name if solutionr.name != 'results' else solutionr.parent.name
    #assert len(results_l_df) == len(results_r_df)

    #compare_df = results_l_df.merge(right=results_r_df, on='name', how='inner', suffixes= (Path(solutionl).parent.name, Path(solutionr).parent.name))
    compare_df = results_l_df.append(results_r_df)
    sns.lineplot(x='run', y='NodeCount', data=compare_df, hue='version')
    plt.show()
    sns.boxplot(x='version', y='NodeCount', data=compare_df[compare_df['NodeCount'] > 1])
    plt.show()
    sns.boxplot(x='version', y='Runtime', data=compare_df[compare_df['NodeCount'] > 1])
    plt.show()

@cli.command()
@click.argument('solution', type=click.Path(exists=True, dir_okay=True, file_okay=False), nargs=-1)
def analyze_results(solution: List[Path]):
    if len(solution) == 0:
        solution = [x for x in Path('results').rglob('*_branching-bench')]
    results_df = create_result_df(solution, re.compile("tiny_(?P<fleet_size>\d+)v_(?P<tours>\d+)t_(?P<num_chargers>\d+)c_(?P<energy_price_variance>\d+\.\d+)epv_run-(?P<run>\d+)_(?P<suffix>.*)"))
    results_df = results_df[~results_df['infeasible'] & ~results_df['unsolved']]
    most_branching = results_df.sort_values(by='NodeCount', ascending=False)
    with pd.option_context('display.max_rows', None):
        print(most_branching[['name', 'NodeCount', 'Runtime']].head(n=200))
    sns.lineplot(x='energy_price_variance', y='NodeCount', data=results_df, sort=True)
    plt.show()
    sns.lineplot(x='energy_price_variance', y='IterCount', data=results_df, sort=True)
    plt.show()
    sns.lineplot(x='energy_price_variance', y='Runtime', data=results_df, sort=True)
    plt.show()
    sns.pointplot(x='run', y='NodeCount', hue='energy_price_variance', data=results_df)
    plt.show()
    sns.pointplot(x='run', y='Runtime', hue='energy_price_variance', data=results_df)
    plt.show()

if __name__ == '__main__':
    cli()
