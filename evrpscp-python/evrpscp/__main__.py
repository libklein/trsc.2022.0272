import argparse
from os import access, W_OK
from pathlib import Path
from typing import Callable
from . import parseInstance

def parsePath(arg: str, mode='r', type='d') -> Path:
    p = Path(arg)
    if not p.exists():
        if type == 'd':
            if mode == 'r':
                raise argparse.ArgumentTypeError(f"Path {p} does not exist")
            else:
                # Try creating the directory
                try:
                    p.mkdir()
                except:
                    raise argparse.ArgumentTypeError(f"Cannot write to {p} (Does not exist)")
        elif type == 'f':
            raise argparse.ArgumentTypeError(f"Path {p} does not exist")

    if type == 'd':
        if not p.is_dir():
            raise argparse.ArgumentTypeError(f"Path {p} is not a directory")
        if mode == 'w' and not access(p, W_OK):
            raise argparse.ArgumentTypeError(f'Path {p} is not writeable')
    return p


def PathParser(mode='r', type='d') -> Callable:
    return lambda arg: parsePath(arg, mode=mode, type=type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse EVRP-SCP instances')
    parser.add_argument('instance', type=PathParser('r', 'f'), help='Instance dump')

    args = parser.parse_args()

    instance = parseInstance(args.instance)
    print(instance)
