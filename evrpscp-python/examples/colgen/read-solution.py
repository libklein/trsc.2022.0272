import click
from pathlib import Path
import pickle


def read_solution(path: Path):
    if path.suffix != '.pickle':
        raise ValueError("Please provide the path to the pickled solution.")
    with path.open('rb') as filestream:
        return pickle.load(filestream)


@click.command("Read solution")
@click.argument("path", type=click.Path(exists=True))
def cli(path: Path):
    """Reads a pickled solution file from PATH."""
    print(read_solution(Path(path)))

if __name__ == '__main__':
    cli()
