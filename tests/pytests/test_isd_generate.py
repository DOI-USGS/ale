#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module has unit tests for the isd_generate functions."""

# This is free and unencumbered software released into the public domain.
#
# The authors of ale do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import unittest
from unittest.mock import call, patch

import ale.isd_generate as isdg


class TestFile(unittest.TestCase):

    @patch("ale.isd_generate.Path.write_text")
    def test_file_to_isd(self, m_path_wt):

        json_text = "some json text"
        cube_str = "dummy.cub"

        with patch("ale.loads", return_value=json_text) as m_loads:
            cube_str = "dummy.cub"
            isdg.file_to_isd(cube_str)
            self.assertEqual(
                m_loads.call_args_list, [call(cube_str, verbose=True)]
            )
            self.assertEqual(
                m_path_wt.call_args_list, [call(json_text)]
            )

        m_path_wt.reset_mock()
        with patch("ale.loads", return_value=json_text) as m_loads:
            out_str = "dummy.json"
            kernel_val = ["list of kernels"]
            isdg.file_to_isd(cube_str, out=out_str, kernels=kernel_val)
            self.assertEqual(
                m_loads.call_args_list,
                [call(cube_str, props={'kernels': kernel_val}, verbose=True)]
            )
            self.assertEqual(
                m_path_wt.call_args_list, [call(json_text)]
            )
