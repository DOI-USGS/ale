import pytest
import json
import os

import ale

import ale.isd_generate as isdg

from conftest import get_image_label, get_isd, compare_dicts


def test_compress_decompress():
    label = get_image_label("EN1072174528M")
    isd_str = get_isd("messmdis_isis")

    json_file = os.path.splitext(label)[0] + '.json'

    isdg.write_json_file(isd_str, json_file)

    compressed_file = isdg.compress_isd(json_file)

    decompressed_file = isdg.decompress_isd(compressed_file)

    with open(decompressed_file, 'r') as fp:
        isis_dict = json.load(fp)

    comparison = compare_dicts(isis_dict, isd_str)
    assert comparison == []


