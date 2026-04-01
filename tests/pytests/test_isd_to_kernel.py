import json
import pytest
import re

from ale.isd_to_kernel import isd_to_kernel
from conftest import get_isd, get_isd_path
from unittest.mock import patch


@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_spk_generation(mock_write_spk, mock_translate, mock_get_name, mock_search, tmp_path):
    """Test that isd_to_kernel correctly handles SPK generation."""
    
    mock_get_name.return_value = "mex"
    mock_search.return_value = [None, {"sclk": ["sclk.tsc"], "lsk": ["lsk.tls"]}]
    mock_translate.return_value = ["MARS", "J2000"]
    
    outfile = tmp_path / "test_spk.bsp"
    
    isd_data = get_isd("ctx")
    isd_file = get_isd_path("ctx")

    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )
    
    assert mock_write_spk.called
    args, kwargs = mock_write_spk.call_args
    
    assert args[0] == str(outfile)                                              # output file path
    assert args[1][0] == isd_data["instrument_position"]["positions"][0]        # state positions
    assert args[2][0] == isd_data["instrument_position"]["ephemeris_times"][0]  # ephemeris times
    assert args[3] == isd_data["naif_keywords"]["BODY_CODE"]                    # body code
    assert args[4] == isd_data["naif_keywords"]["BODY_FRAME_CODE"]              # body frame code
    assert args[5] == "J2000"                                                   # reference frame
    assert args[6] == f"{mock_get_name.return_value}:{isd_data['name_sensor']}" # segment id
    assert args[7] == 1                                                         # degree
    assert args[8][0] == isd_data["instrument_position"]["velocities"][0]       # state velocities
    assert "USGS ALE Generated SPK Kernel" in args[9]                           # comment header

    assert len(args[1]) == len(args[2]) == len(args[8]) == 401


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("ale.isd_to_kernel.write_ck")
def test_ck_generation(mock_write_ck, mock_translate, mock_search, mock_get_name, tmp_path):
    """Test that isd_to_kernel correctly handles CK generation."""
    
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    
    # Mock return for SCLK and LSK search
    mock_search.return_value = [None, {
        "sclk": ["mex_sclk.tsc"],
        "lsk": ["naif0012.tls"]
    }] 
    
    outfile = tmp_path / "test_ck.bc"

    isd_data = get_isd("ctx")
    isd_file = get_isd_path("ctx")
    
    isd_to_kernel(
        isd_file=isd_file,
        kernel_type="ck",
        outfile=outfile,
        overwrite=True
    )
    
    assert mock_write_ck.called
    args, kwargs = mock_write_ck.call_args
    
    assert args[0] == str(outfile)                                                  # output file path
    assert args[1][0] == isd_data["instrument_pointing"]["quaternions"][0]          # quaternions
    assert args[2][0] == isd_data["instrument_pointing"]["ephemeris_times"][0]      # ephemeris times
    assert args[3] == isd_data["instrument_pointing"]["time_dependent_frames"][0]   # instrument frame code
    assert args[4] == "J2000"                                                       # reference frame
    assert args[6] == ["mex_sclk.tsc"]                                              # sclk kernels list
    assert args[7] == "naif0012.tls"                                                # lsk kernel (first element of list)
    assert args[8][0] == isd_data["instrument_pointing"]["angular_velocities"][0]   # angular velocities
    assert "USGS ALE Generated CK Kernel" in args[9]                                # comment header

    assert len(args[1]) == len(args[2]) == len(args[8]) == 401


@patch("pyspiceql.writeTextKernel")
def test_text_kernel_generation(mock_write_text, tmp_path):
    """Test that isd_to_kernel correctly handles text kernel generation."""
    
    kernel_type = "IK"
    outfile = tmp_path / "test.ti"
    data = '{"TEST_KEYWORD": "TEST_VALUE"}'

    isd_to_kernel(
        kernel_type=kernel_type,
        data=data,
        outfile=outfile
    )
    
    assert mock_write_text.called
    args, kwargs = mock_write_text.call_args
    
    assert args[0] == str(outfile)
    assert args[1] == kernel_type
    assert args[2] == json.loads(data)


def test_invalid_isd_extension():
    """Verify that non-JSON files raise an error."""
    expected_msg = "ISD must be in JSON"
    with pytest.raises(Exception, match=expected_msg):
        isd_to_kernel(isd_file="test.txt", kernel_type="spk")


def test_invalid_kernel_type():
    """Verify that invalid kernel types raise an error."""
    expected_msg = "Kernel type [abc] is not valid. Choose from the following: ['SPK', 'CK', 'FK', 'IK', 'LSK', 'MK', 'PCK', 'SCLK']"
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


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_outfile_extension_correction(mock_write_spk, mock_translate, mock_search, mock_get_name, tmp_path):
    """Verify that isd_to_kernel corrects a wrong extension (e.g., .txt -> .bsp)."""
    
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]
    
    outfile = tmp_path / "test.abc"
    expected_outfile = str(tmp_path / "test.bsp")
    
    isd_to_kernel(
        isd_file=get_isd_path("ctx"),
        kernel_type="spk",
        outfile=outfile,
        overwrite=True
    )
    
    # The function should have changed 'test.abc' to 'test.bsp'
    args, _ = mock_write_spk.call_args
    actual_path_used = args[0]
    
    assert actual_path_used == expected_outfile
    assert actual_path_used.endswith(".bsp")
    assert not actual_path_used.endswith(".abc")


@patch("pyspiceql.getSpiceqlName")
@patch("pyspiceql.searchForKernelsets")
@patch("pyspiceql.translateCodeToName")
@patch("pyspiceql.writeSpk")
def test_mismatched_times_positions(mock_write, mock_translate, mock_search, mock_get_name, tmp_path):
    """Verify state positions and times size are same."""
    mock_get_name.return_value = "mex"
    mock_translate.return_value = ["MARS", "J2000"]
    mock_search.return_value = [None, {"sclk": ["mock.tsc"], "lsk": ["mock.tls"]}]
    
    isd_data = get_isd("ctx")

    # Bump only ephemeris times
    isd_data["instrument_position"]["ephemeris_times"].append(9999.0)
    broken_isd = tmp_path / "bad.json"
    broken_isd.write_text(json.dumps(isd_data))

    with pytest.raises(ValueError, match="Positions and Times length mismatch!"):
        isd_to_kernel(isd_file=broken_isd, kernel_type="spk")
