from packaging.version import Version


def test_import():
    import yaunet

    Version(yaunet.__version__)
