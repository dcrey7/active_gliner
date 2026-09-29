"""O1 smoke tests: the package installs, imports, and exposes its CLI."""

import subprocess
import sys

import active_gliner


def test_package_has_version():
    assert isinstance(active_gliner.__version__, str)
    assert active_gliner.__version__


def test_cli_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "active_gliner", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert active_gliner.__version__ in result.stdout


def test_gliner2_is_importable():
    import gliner2

    assert hasattr(gliner2, "GLiNER2") or hasattr(gliner2, "AutoExtractor")
