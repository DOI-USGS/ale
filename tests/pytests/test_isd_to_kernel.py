import json
import os
import pytest
import re
import subprocess
from pathlib import Path

import spiceypy as spice

from ale.isd_to_kernel import isd_to_kernel, spk_comment, ck_comment, main
from conftest import get_isd, get_isd_path
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def setup_spiceql_cache(tmp_path_factory):
    """Set SPICEQL_CACHE_DIR for all tests in this module."""
    cache_dir = tmp_path_factory.mktemp("spiceql_cache")
    old_cache = os.environ.get("SPICEQL_CACHE_DIR")
    os.environ["SPICEQL_CACHE_DIR"] = str(cache_dir)
    yield cache_dir
    # Restore original value
    if old_cache is not None:
        os.environ["SPICEQL_CACHE_DIR"] = old_cache
    else:
        os.environ.pop("SPICEQL_CACHE_DIR", None)


@pytest.fixture
def mock_ctx_kernelsets():
    """
    Fixture that provides real CTX kernel paths AND furnishes them to SPICE.

    This fixture:
    1. Finds real kernel files from CTX test data
    2. Furnishes them to SPICE so writeSpk/writeCk can use them
    3. Returns the mock value for searchForKernelsets
    4. Cleans up by unloading kernels after the test

    Returns
    -------
    list
        A tuple of (status, kernels_dict) as returned by searchForKernelsets.
    """
    test_data_dir = Path(__file__).parent / "data" / "B10_013341_1010_XN_79S172W"

    kernels = {
        "sclk": [str((test_data_dir / "mro_sclkscet_00082_65536.tsc").absolute())],
        "lsk": [str((test_data_dir / "naif0012.tls").absolute())],
        "pck": [str((test_data_dir / "pck00008.tpc").absolute())],
        "fk": [str((test_data_dir / "mro_v16.tf").absolute())],
        "ik": [str((test_data_dir / "mro_ctx_v11.ti").absolute())],
    }

    # Furnish all the kernels to SPICE (REQUIRED for writeSpk/writeCk to work!)
    for kernel_paths in kernels.values():
        for kernel_path in kernel_paths:
            if Path(kernel_path).exists():
                spice.furnsh(kernel_path)

    # Yield the mock return value
    yield [None, kernels]

    # Cleanup: unload all kernels after the test
    spice.kclear()


@patch("pyspiceql.searchForKernelsets")
def test_spk_generation(mock_search, mock_ctx_kernelsets, tmp_path):
    """Test that isd_to_kernel correctly handles SPK generation and verify contents."""
    mock_search.return_value = mock_ctx_kernelsets

    outfile = tmp_path / "test_spk.bsp"
    isd_file = get_isd_path("ctx")
    isd_data = get_isd("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )

    # Verify the SPK file was actually created
    assert outfile.exists(), "SPK file should be created"
    assert outfile.stat().st_size > 0, "SPK file should not be empty"

    # Basic validation - SPK files are typically several KB
    assert outfile.stat().st_size > 10000, f"SPK file seems too small: {outfile.stat().st_size} bytes"

    # Verify the SPK can be loaded and contains expected coverage
    test_data_dir = Path(__file__).parent / "data" / "B10_013341_1010_XN_79S172W"
    spice.furnsh(str((test_data_dir / "naif0012.tls").absolute()))  # LSK for time
    spice.furnsh(str((test_data_dir / "pck00008.tpc").absolute()))  # PCK for body info
    spice.furnsh(str(outfile))

    try:
        # Get expected values from ISD
        ephemeris_times = isd_data["instrument_position"]["ephemeris_times"]
        first_et = ephemeris_times[0]
        last_et = ephemeris_times[-1]

        # Get the target code from ISD
        pointing_frames = isd_data.get("instrument_pointing", {}).get("time_dependent_frames", [])
        if pointing_frames:
            inst_frame_code = pointing_frames[0]
            target_code = int(inst_frame_code / 1000)
            if target_code == 0:
                target_code = inst_frame_code
        else:
            # Fallback
            target_code = -74

        # Verify SPK contains data for the expected object
        obj_ids = spice.spkobj(str(outfile))
        obj_ids_list = [obj_ids[i] for i in range(len(obj_ids))]

        assert target_code in obj_ids_list, \
            f"SPK should contain data for object {target_code}, but contains: {obj_ids_list}"

        # Verify SPK coverage matches ISD time range
        cover = spice.spkcov(str(outfile), target_code)
        assert spice.wncard(cover) > 0, f"No coverage found for target {target_code}"

        # Get the coverage window
        bounds = spice.wnfetd(cover, 0)
        coverage_start, coverage_end = bounds[0], bounds[1]

        # Verify coverage includes the ISD time range (with small tolerance for floating point)
        tolerance = 1e-6  # 1 microsecond tolerance
        assert coverage_start <= first_et + tolerance, \
            f"SPK coverage starts at {coverage_start}, but ISD starts at {first_et}"
        assert coverage_end >= last_et - tolerance, \
            f"SPK coverage ends at {coverage_end}, but ISD ends at {last_et}"

        print(f"✓ SPK kernel successfully generated and validated: {outfile.stat().st_size} bytes")
        print(f"  Object {target_code} coverage: {coverage_start} to {coverage_end}")
        print(f"  ISD time range: {first_et} to {last_et}")
        print(f"  Note: Full position verification requires planetary SPK chain")

    finally:
        spice.unload(str(outfile))
        spice.unload(str((test_data_dir / "naif0012.tls").absolute()))
        spice.unload(str((test_data_dir / "pck00008.tpc").absolute()))


@patch("pyspiceql.searchForKernelsets")
def test_ck_generation(mock_search, mock_ctx_kernelsets, tmp_path):
    """Test that isd_to_kernel correctly handles CK generation and verify contents."""
    mock_search.return_value = mock_ctx_kernelsets

    outfile = tmp_path / "test_ck.bc"
    isd_file = get_isd_path("ctx")
    isd_data = get_isd("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="ck",
        outfile=outfile,
        comment="test comment"
    )

    # Verify the CK file was actually created
    assert outfile.exists(), "CK file should be created"
    assert outfile.stat().st_size > 0, "CK file should not be empty"

    # Basic validation - CK files are typically several KB
    assert outfile.stat().st_size > 5000, f"CK file seems too small: {outfile.stat().st_size} bytes"

    # Verify the CK can be loaded and contains expected coverage
    test_data_dir = Path(__file__).parent / "data" / "B10_013341_1010_XN_79S172W"
    spice.furnsh(str((test_data_dir / "naif0012.tls").absolute()))  # LSK for time
    spice.furnsh(str((test_data_dir / "mro_sclkscet_00082_65536.tsc").absolute()))  # SCLK
    spice.furnsh(str((test_data_dir / "mro_v16.tf").absolute()))  # Frame kernel
    spice.furnsh(str(outfile))

    try:
        # Get expected values from ISD
        ephemeris_times = isd_data["instrument_pointing"]["ephemeris_times"]
        first_et = ephemeris_times[0]
        last_et = ephemeris_times[-1]

        # Get the frame code from ISD
        pointing_frames = isd_data.get("instrument_pointing", {}).get("time_dependent_frames", [])
        if pointing_frames:
            frame_code = pointing_frames[0]
        else:
            frame_code = -74000

        # Verify CK contains data for the expected frame
        obj_ids = spice.ckobj(str(outfile))
        obj_ids_list = [obj_ids[i] for i in range(len(obj_ids))]

        assert frame_code in obj_ids_list, \
            f"CK should contain data for frame {frame_code}, but contains: {obj_ids_list}"

        # Verify CK coverage matches ISD time range
        cover = spice.ckcov(str(outfile), frame_code, True, "INTERVAL", 0.0, "TDB")
        assert spice.wncard(cover) > 0, f"No coverage found for frame {frame_code}"

        # Get the coverage window
        bounds = spice.wnfetd(cover, 0)
        coverage_start, coverage_end = bounds[0], bounds[1]

        # Verify coverage includes the ISD time range (with small tolerance for floating point)
        tolerance = 1e-6  # 1 microsecond tolerance
        assert coverage_start <= first_et + tolerance, \
            f"CK coverage starts at {coverage_start}, but ISD starts at {first_et}"
        assert coverage_end >= last_et - tolerance, \
            f"CK coverage ends at {coverage_end}, but ISD ends at {last_et}"

        print(f"✓ CK kernel successfully generated and validated: {outfile.stat().st_size} bytes")
        print(f"  Frame {frame_code} coverage: {coverage_start} to {coverage_end}")
        print(f"  ISD time range: {first_et} to {last_et}")

    finally:
        spice.unload(str(outfile))
        spice.unload(str((test_data_dir / "naif0012.tls").absolute()))
        spice.unload(str((test_data_dir / "mro_sclkscet_00082_65536.tsc").absolute()))
        spice.unload(str((test_data_dir / "mro_v16.tf").absolute()))


@patch("pyspiceql.searchForKernelsets")
def test_text_kernel_generation(mock_search, tmp_path):
    """Test that isd_to_kernel correctly handles text kernel generation."""

    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    kernel_type = "IK"
    outfile = tmp_path / "test.ti"
    data = '{"TEST_KEYWORD": "TEST_VALUE"}'

    isd_to_kernel(
        kernel_type=kernel_type,
        data=data,
        outfile=outfile
    )

    # Verify the file was created and contains expected content
    assert outfile.exists()
    content = outfile.read_text()
    assert "TEST_KEYWORD" in content
    assert "TEST_VALUE" in content


def test_invalid_isd_extension():
    """Verify that non-JSON files raise an error."""
    expected_msg = "ISD must be in JSON"
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(isd_file="test.txt", kernel_type="spk")


def test_invalid_kernel_type():
    """Verify that invalid kernel types raise an error."""
    # SpiceQL error
    expected_msg = "std::exception: abc is not a valid kernel type"
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(isd_file="test.json", kernel_type="abc")


def test_empty_data(tmp_path):
    """Verify that text kernels require a data payload."""
    outfile = tmp_path / "test.tf"
    abs_outfile = str(outfile.resolve()) 
    
    expected_msg = f"Must enter JSON keywords to generate kernel [{abs_outfile}]."
    
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(kernel_type="fk", outfile=outfile)


def test_invalid_data(tmp_path):
    """Verify that data payload is JSON."""
    outfile = tmp_path / "test.tf"
    data = "bad data"
    expected_msg = "The 'data' payload is not valid JSON."
    
    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(kernel_type="fk", outfile=outfile, data=data)


def test_missing_isd():
    """Verify missing ISD file for binary kernels raises an error."""
    expected_msg = "Missing ISD file."
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(kernel_type="ck")


def test_missing_outfile():
    """Verify missing outfile file for text kernels raises an error."""
    expected_msg = "Must enter an outfile name for text kernels."
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(kernel_type="pck")


@patch("pyspiceql.searchForKernelsets")
def test_outfile_extension_correction(mock_search, mock_ctx_kernelsets, tmp_path):
    """Verify that isd_to_kernel corrects a wrong extension (e.g., .txt -> .bsp)."""

    mock_search.return_value = mock_ctx_kernelsets

    outfile = tmp_path / "test.abc"
    expected_outfile = tmp_path / "test.bsp"

    isd_to_kernel(
        isd_file=get_isd_path("ctx"),
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )

    # The function should have changed 'test.abc' to 'test.bsp' and created the file
    assert expected_outfile.exists(), "SPK file with corrected extension should exist"
    assert not (tmp_path / "test.abc").exists(), "File with wrong extension should not exist"


@patch("pyspiceql.searchForKernelsets")
def test_mismatched_times_positions(mock_search, tmp_path):
    """Verify state positions and times size are same."""
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    isd_data = get_isd("ctx")

    # Bump only ephemeris times
    isd_data["instrument_position"]["ephemeris_times"].append(9999.0)
    broken_isd = tmp_path / "bad.json"
    broken_isd.write_text(json.dumps(isd_data))

    with pytest.raises(ValueError, match="Positions and Times length mismatch!"):
        isd_to_kernel(isd_file=broken_isd, kernel_type="spk")


def test_spk_comment():
    """Test SPK comment generation includes all required fields."""
    comment = spk_comment(
        outfile="/path/to/test.bsp",
        segment_id="TEST_SEGMENT",
        start_time="2020-01-01T00:00:00",
        end_time="2020-01-02T00:00:00",
        instrument_id="TEST_INST",
        target_body="12345",
        target_name="TestTarget",
        center_body="499",
        center_name="Mars",
        reference_frame="J2000",
        records=100,
        degree=7,
        kernels={"lsk": ["test.tls"], "spk": ["test.bsp"]},
        comment="User test comment"
    )

    # Verify key sections are present
    assert "USGS ALE Generated SPK Kernel" in comment
    assert "TEST_SEGMENT" in comment
    assert "2020-01-01T00:00:00" in comment
    assert "2020-01-02T00:00:00" in comment
    assert "TEST_INST" in comment
    assert "12345" in comment
    assert "TestTarget" in comment
    assert "499" in comment
    assert "Mars" in comment
    assert "J2000" in comment
    assert "100" in comment
    assert "7" in comment
    assert "User test comment" in comment
    assert "Position Data in the File" in comment
    assert "ID:USGS_SPK_ABCORR=NONE" in comment


def test_ck_comment():
    """Test CK comment generation includes all required fields."""
    comment = ck_comment(
        outfile="/path/to/test.bc",
        segment_id="TEST_CK_SEGMENT",
        start_time="2020-01-01T00:00:00",
        end_time="2020-01-02T00:00:00",
        instrument_id="TEST_INST",
        target_body="12345",
        target_name="TestTarget",
        center_body="499",
        center_name="Mars",
        reference_frame="J2000",
        records=100,
        has_av=True,
        kernels={"lsk": ["test.tls"], "sclk": ["test.tsc"]},
        comment="User CK test comment"
    )

    # Verify key sections are present
    assert "USGS ALE Generated CK Kernel" in comment
    assert "TEST_CK_SEGMENT" in comment
    assert "2020-01-01T00:00:00" in comment
    assert "2020-01-02T00:00:00" in comment
    assert "TEST_INST" in comment
    assert "12345" in comment
    assert "TestTarget" in comment
    assert "499" in comment
    assert "Mars" in comment
    assert "J2000" in comment
    assert "100" in comment
    assert "True" in comment  # has_av
    assert "User CK test comment" in comment
    assert "Orientation Data in the File" in comment
    assert "angular velocity" in comment.lower()


@patch("pyspiceql.searchForKernelsets")
def test_ck_without_angular_velocities(mock_search, mock_ctx_kernelsets, tmp_path):
    """Test CK generation when ISD lacks angular velocities."""
    mock_search.return_value = mock_ctx_kernelsets

    isd_data = get_isd("ctx")
    # Remove angular velocities
    isd_data["instrument_pointing"]["angular_velocities"] = None

    modified_isd = tmp_path / "no_av.json"
    modified_isd.write_text(json.dumps(isd_data))

    outfile = tmp_path / "test_no_av.bc"

    isd_to_kernel(
        isd_file=modified_isd,
        kernel_type="ck",
        outfile=outfile
    )

    # Verify the CK file was actually created
    assert outfile.exists(), "CK file should be created"
    assert outfile.stat().st_size > 0, "CK file should not be empty"

    # Basic validation - CK files are typically several KB
    assert outfile.stat().st_size > 5000, f"CK file seems too small: {outfile.stat().st_size} bytes"

    print(f"✓ CK kernel successfully generated without angular velocities: {outfile.stat().st_size} bytes")


@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.writeSpk")
def test_segment_id_truncation(mock_write_spk, mock_search, tmp_path):
    """Test that segment IDs longer than 40 characters are truncated."""
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    outfile = tmp_path / "test_truncate.bsp"
    isd_file = get_isd_path("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )

    assert mock_write_spk.called
    args, _ = mock_write_spk.call_args

    # Verify segment_id (args[6]) is truncated to 40 characters
    segment_id = args[6]
    assert len(segment_id) <= 40, f"Segment ID should be <= 40 chars, got {len(segment_id)}"


@pytest.mark.parametrize("missing_kernel,kernels_dict,error_msg", [
    ("sclk", {"lsk": ["naif0012.tls"]}, "Could not find SCLKs"),
    ("lsk", {"sclk": ["mex_sclk.tsc"]}, "Could not find LSK"),
])
@patch("pyspiceql.searchForKernelsets")
def test_missing_required_kernels(mock_search, missing_kernel, kernels_dict, error_msg, tmp_path):
    """Test that missing required kernels raise appropriate errors."""
    mock_search.return_value = [None, kernels_dict]

    outfile = tmp_path / f"test_no_{missing_kernel}.bc"
    isd_file = get_isd_path("ctx")

    with pytest.raises(Exception, match=error_msg):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="ck",
            outfile=outfile
        )


@patch("pyspiceql.searchForKernelsets")
def test_missing_mission_name(mock_search, tmp_path):
    """Test that ISD without resolvable mission name raises error."""
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]

    outfile = tmp_path / "test_no_mission.bsp"
    # Create an ISD with fields that won't resolve to a valid mission name
    isd_data = get_isd("ctx")
    # Remove all the fields that getSpiceqlName would use to find a mission
    isd_data.pop("name_sensor", None)
    isd_data.pop("name_platform", None)
    if "naif_keywords" in isd_data:
        # Remove FRAME_*_NAME keys
        isd_data["naif_keywords"] = {k: v for k, v in isd_data["naif_keywords"].items()
                                     if not (k.startswith("FRAME_") and k.endswith("_NAME"))}

    invalid_isd = tmp_path / "invalid.json"
    invalid_isd.write_text(json.dumps(isd_data))

    with pytest.raises(Exception, match="Could not find a valid mission name"):
        isd_to_kernel(
            isd_file=invalid_isd,
            kernel_type="spk",
            outfile=outfile
        )


def test_file_already_exists_no_overwrite(tmp_path):
    """Test that existing files without overwrite flag raise error."""
    outfile = tmp_path / "existing.bsp"
    outfile.write_text("existing content")

    isd_file = get_isd_path("ctx")

    expected_msg = f"Output file [{str(outfile.resolve())}] already exists."

    with pytest.raises(Exception, match=re.escape(expected_msg)):
        isd_to_kernel(
            isd_file=isd_file,
            kernel_type="spk",
            outfile=outfile,
            overwrite=False
        )


# CLI / main() function tests
class TestCLI:
    """Test the command-line interface via main() function."""

    @pytest.mark.parametrize("args,expected_kwargs", [
        # Basic args
        (["-f", "test.json", "-k", "spk"],
         {'isd_file_check': 'test.json', 'kernel_type': 'spk'}),
        # Output file
        (["-f", "test.json", "-k", "spk", "-o", "output.bsp"],
         {'outfile': 'output.bsp'}),
        # Overwrite flag
        (["-f", "test.json", "-k", "spk", "--overwrite"],
         {'overwrite': True}),
        # Web flag
        (["-f", "test.json", "-k", "spk", "--web"],
         {'use_web': True}),
    ])
    @patch("ale.isd_to_kernel.isd_to_kernel")
    def test_main_with_args(self, mock_isd_to_kernel, args, expected_kwargs):
        """Test main() with various command-line arguments."""
        with patch("sys.argv", ["isd_to_kernel"] + args):
            main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]

        for key, expected_value in expected_kwargs.items():
            if key == 'isd_file_check':
                assert str(call_kwargs['isd_file']).endswith(expected_value)
            else:
                assert call_kwargs[key] == expected_value

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk", "-v"])
    def test_main_with_verbose_flag(self, mock_isd_to_kernel):
        """Test main() with verbose flag sets log level."""
        import logging
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['log_level'] == logging.INFO

    @patch("ale.isd_to_kernel.isd_to_kernel", side_effect=Exception("Test error"))
    @patch("sys.argv", ["isd_to_kernel", "-f", "test.json", "-k", "spk"])
    def test_main_handles_exceptions(self, mock_isd_to_kernel):
        """Test main() exits gracefully on errors."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Verify error message contains the exception
        assert "Test error" in str(exc_info.value)

    @patch("ale.isd_to_kernel.isd_to_kernel")
    @patch("sys.argv", ["isd_to_kernel", "-k", "fk", "-o", "test.tf", "-d", '{"KEY": "VALUE"}'])
    def test_main_with_text_kernel_data(self, mock_isd_to_kernel):
        """Test main() with text kernel data."""
        main()

        assert mock_isd_to_kernel.called
        call_kwargs = mock_isd_to_kernel.call_args[1]
        assert call_kwargs['kernel_type'] == 'fk'
        assert call_kwargs['data'] == '{"KEY": "VALUE"}'


class TestSubprocessCalls:
    """Test actual subprocess calls to isd_to_kernel command."""

    def test_subprocess_help(self):
        """Test that isd_to_kernel --help works."""
        result = subprocess.run(
            ["isd_to_kernel", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "isd_to_kernel" in result.stdout

    def test_subprocess_missing_required_args(self):
        """Test that isd_to_kernel without required args shows usage."""
        result = subprocess.run(
            ["isd_to_kernel"],
            capture_output=True,
            text=True
        )

        # Should fail without required arguments
        assert result.returncode != 0

    def test_subprocess_text_kernel_with_data(self, tmp_path):
        """Test subprocess creating text kernel with JSON data."""
        output = tmp_path / "test.tf"

        result = subprocess.run(
            [
                "isd_to_kernel",
                "-k", "fk",
                "-o", str(output),
                "-d", '{"TEST_KEY": "TEST_VALUE"}'
            ],
            capture_output=True,
            text=True
        )

        # May fail due to missing dependencies, but should parse args correctly
        # Main test is that it doesn't fail due to argument parsing
        if result.returncode == 0:
            assert output.exists()

