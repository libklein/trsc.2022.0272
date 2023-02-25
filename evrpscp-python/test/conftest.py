# coding=utf-8
import pytest

from evrpscp import Dump, DiscretizedInstance

@pytest.fixture
def instance(shared_datadir):
    for inst_file in shared_datadir.glob('instances/*.dump.d'):
        return DiscretizedInstance.DiscretizeInstance(Dump.ParseDump(inst_file).instance)