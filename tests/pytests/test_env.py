import os

import pytest
import warnings
from importlib import reload

from unittest.mock import patch

import ale

def test_env_not_set():
    assert not ale.spice_root

def test_env_set():
    with patch.dict('os.environ', {'ALESPICEROOT': '/foo/bar'}):
        reload(ale)
        assert ale.spice_root == '/foo/bar'
    reload(ale)
    assert not ale.spice_root

