import toml
from typing import List, Dict, Optional
from pathlib import Path
from funcy import cached_readonly, first
from dataclasses import dataclass, field

from evrpscp import SchedulingInstance
from .parsing import *

class InstanceRepresentation:
    TAG = ''

    def __init__(self, filename: Path):
        self.filename = filename

    def parse(self):
        raise NotImplementedError(f'{self.TAG} does not implement parse() yet!')

class JSONInstanceRepresentation(InstanceRepresentation):
    TAG = 'json'

    def parse(self):
        return parseInstance(self.filename)


class XMLInstanceRepresentation(InstanceRepresentation):
    TAG = 'xml'
    pass

class PelletierInstanceRepresentation(InstanceRepresentation):
    TAG = 'pelletier'

    def parse(self):
        return parsePelletier(self.filename, self.filename.parent / 'Parameters')
        pass


class SolutionRepresentation:
    TAG = ''

    def __init__(self, filename: Path):
        self.filename = filename

    def parse(self):
        raise NotImplementedError(f'{self.TAG} does not implement parse() yet!')


class MIPSolutionRepresentation(SolutionRepresentation):
    TAG = 'mst'
    pass

class PythonSolutionRepresentation(SolutionRepresentation):
    TAG = 'python'

    def parse(self):
        return parseSolution(self.filename)


@dataclass
class Dump:
    """
    Dump can have
        - instance file (multiple representations).
        - Period solution pool
        - Routing information
    """
    directory: Path
    name: str
    instance_representations: Dict[str, InstanceRepresentation] = field(default_factory=dict)
    solution_representations: Dict[str, SolutionRepresentation] = field(default_factory=dict)
    period_solution_pool: Optional[List[Path]] = None
    fixed_tour_schedule: Optional[Path] = None

    @staticmethod
    def DumpSchedulingInstance(directory: Path, instance_name: str, instance: SchedulingInstance, is_discretized=False):
        """
        directory -- The root directory of the dump
        instance_name -- The name of the instance
        instance -- The instance object
        """
        if not is_discretized:
            raise NotImplementedError("Non-discrete pelletier dumps are not yet implemented!")
        # Create manifest
        manifest = {
            'instance': {
                'name': f'{instance_name}',
                'representations': {
                    'json': {
                        'filename': f'{instance_name}.json',
                    }
                }
            }
        }
        if not directory.exists():
            directory.mkdir()

        with open(str(directory / 'manifest.toml'), 'w') as manifest_file:
            toml.dump(manifest, manifest_file)

        # Write instance file
        with open(str(directory / f'{instance_name}.json'), 'w') as instance_file:
            instance_file.write(instance.to_json())

    @cached_readonly
    def instance(self):
        for x in self.instance_representations.values():
            x.parse()
            try:
                return x.parse()
            except Exception as e:
                print(e)
        raise NotImplementedError(f'Could not parse any of the instance representations : {", ".join(self.instance_representations.keys())}')

    @cached_readonly
    def solution(self):
        for x in self.solution_representations.values():
            try:
                return x.parse()
            except Exception as e:
                pass
        raise NotImplementedError(f'Could not parse any of the solution representations : {", ".join(self.solution_representations.keys())}')

    @cached_readonly
    def solution_pool(self):
        if not self.period_solution_pool:
            raise ValueError("Dump contains no solution pool")
        return parseSolutionPool([(self.directory / x) for x in self.period_solution_pool])

    def add_instance_representation(self, tag: str, filename: Path):
        for x in InstanceRepresentation.__subclasses__():
            if x.TAG == tag:
                self.instance_representations[tag] = x(self.directory / filename)

    def add_solution_representation(self, tag: str, filename: Path):
        for x in SolutionRepresentation.__subclasses__():
            if x.TAG == tag:
                self.solution_representations[tag] = x(self.directory / filename)

    @staticmethod
    def ParseDump(path: Path):
        if not (path / "manifest.toml").exists():
            raise ValueError(f"Path {path} is not a valid dump!")
        manifest = toml.load(path / "manifest.toml")
        dump = Dump(directory=path, name=manifest["instance"]["name"])

        for inst_repr, desc in manifest["instance"]["representations"].items():
            try:
                dump.add_instance_representation(inst_repr, desc["filename"])
            except:
                pass

        if "solution" not in manifest:
            return dump

        try:
            dump.period_solution_pool = manifest["solution"]["routing-solution-pool"]["files"]
        except:
            pass

        if "mip-solution" in manifest["solution"]:
            for sol_repr, data in manifest["solution"]["mip-solution"]["representations"].items():
                try:
                    dump.add_solution_representation(sol_repr, data["filename"])
                except:
                    pass

        return dump