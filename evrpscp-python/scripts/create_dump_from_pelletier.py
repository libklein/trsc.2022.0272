# coding=utf-8
import argparse
import toml
from pathlib import Path
from evrpscp import PathParser
from shutil import copytree, copy
from os import mkdir, symlink, open as os_open, O_RDONLY, close as os_close

def create_dump(instance_path: Path, param_path: Path, output_path: Path, rates: str):
    # Create dump folder
    instance_name = instance_path.stem
    dump_path = output_path / f'{instance_name}_{rates}.dump.d'
    try:
        mkdir(str(dump_path))
    except FileExistsError:
        pass
    # Create manifest
    manifest = {
        'instance': {
            'name': f'{instance_name}_{rates}',
            'representations': {
                'pelletier': {
                    'filename': f'instance/{instance_path.name}',
                    'param_path': f'instance/Parameters'
                }
            }
        }
    }
    with open(str(dump_path / 'manifest.toml'), 'w') as manifest_file:
        toml.dump(manifest, manifest_file)
    # Copy Parameters and Instance to directory
    try:
        mkdir(str(dump_path / 'instance'))
    except FileExistsError:
        pass
    dump_param_dir = copytree(str(param_path), str(dump_path / 'instance' / 'Parameters'), dirs_exist_ok=True)
    copy(str(instance_path), str(dump_path / 'instance'))
    # Symlink to correct rates
    param_fd = os_open(dump_param_dir, O_RDONLY)
    try:
        (Path(dump_param_dir) / 'Rates.json').unlink(missing_ok=True)
        symlink(f'{rates}Rates-TOU-EV-4.json', 'Rates.json', dir_fd=param_fd)
        os_close(param_fd)
    except Exception as e:
        os_close(param_fd)
        raise e

    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pelletier instance dump creator')
    parser.add_argument('instances', type=PathParser(mode='r', type='f'), nargs='+',
                        help="Instances to parse")
    parser.add_argument('--param-path', '-p', dest='param_path', type=PathParser(mode='r', type='d'),
                        help='Instance parameter location')
    parser.add_argument('--output-path', '-o', dest='output_path', type=PathParser(mode='w', type='d'), default='-',
                        help='Path were dumps should be written to or \"-\" to save directly to the respective '
                             'parent directories.')
    parser.add_argument('--rates', dest='rates', choices=['Summer', 'Winter'], default='Winter',
                        help='Rate selection')

    arguments = parser.parse_args()

    for i in arguments.instances:
        _output_path = arguments.output_path if arguments.output_path != '-' else i.parent
        create_dump(instance_path=i, param_path=arguments.param_path, output_path=_output_path,
                    rates=arguments.rates)