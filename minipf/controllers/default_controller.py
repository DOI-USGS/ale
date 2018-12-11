# import connexion
# import six

# from minipf.models.data import Data  # noqa: E501
# from minipf.models.isd200 import ISD200  # noqa: E501
# from minipf.models.request_isd import RequestISD  # noqa: E501
from minipf import util
from minipf import drivers

def create_isd(lbl_file):  # noqa: E501
    """Converts Image Labels to ISDs

    Adds an item to the system # noqa: E501

    :param request_isd:
    :type request_isd: dict | bytes

    :rtype: ISD200
    """
    with open(lbl_file, "r") as fp:
        lines = fp.read()

    return drivers.load(lines)

def get_metakernel(mission, year, version):  # noqa: E501
    """Get a specific kernel

     # noqa: E501

    :param mission:
    :type mission: str
    :param year:
    :type year: str
    :param version:
    :type version: str

    :rtype: Data
    """
    if connexion.request.is_json:
        request_isd = RequestISD.from_dict(connexion.request.get_json())  # noqa: E501

    return util.get_metakernels(missions=mission, years=year, versions=version)


def metakernel_catalog():  # noqa: E501
    """Access Product Information

    Get Available Products and Related Metadata # noqa: E501


    :rtype: Data
    """
    return 'do some magic!'

if __name__ == "__main__":
    print("Banana")
