"""Build hook: ship the repository's community extensions (``extensions/``) as the package's catalog.

Everything else is configured in ``pyproject.toml``. A built package carries every extension folder
(those with a ``manifest.json``) as ``wonders/catalog/<id>/``; an editable install reads ``extensions/``
directly (:func:`wonders.extensions.catalog_homes`).
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildWithCatalog(build_py):
    """``build_py`` that also copies ``extensions/*/`` into ``wonders/catalog/``."""

    def run(self):
        super().run()
        source = ROOT / 'extensions'
        if self.editable_mode or not source.is_dir():
            return
        target = Path(self.build_lib) / 'wonders' / 'catalog'
        for folder in sorted(p for p in source.iterdir() if (p / 'manifest.json').is_file()):
            shutil.copytree(folder, target / folder.name, dirs_exist_ok=True)


setup(cmdclass={'build_py': BuildWithCatalog})
