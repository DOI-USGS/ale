import pytest
import warnings
from importlib import reload

import ale

def test_env_not_set(monkeypatch):
    monkeypatch.delenv('ALESPICEROOT', raising=False)
    reload(ale)
    assert not ale.spice_root

def test_env_set(monkeypatch):
    monkeypatch.setenv('ALESPICEROOT', '/foo/bar')
    reload(ale)
    assert ale.spice_root == '/foo/bar'
