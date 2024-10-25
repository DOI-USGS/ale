import pytest
import json
import os

import ale

import ale.isd_generate as isdg

from conftest import get_image_label, get_isd, compare_dicts


def test_compress_decompress():
    label = get_image_label("EN1072174528M")
    isd_str = get_isd("messmdis_isis")

    compressed_file = os.path.splitext(label)[0] + '.br'

    isdg.compress_json(isd_str, compressed_file)

    decompressed_file = isdg.decompress_json(compressed_file)

    with open(decompressed_file, 'r') as fp:
        isis_dict = json.load(fp)

    comparison = compare_dicts(isis_dict, isd_str)
    assert comparison == []


