from importlib.metadata import version

import pinf


def test_package_imports_as_pinf_and_reports_installed_version():
    assert pinf.__version__ == version("pinf")
