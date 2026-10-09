"""Test-wide settings."""

import os

# Installed extensions (~/.wonders/extensions, plugins) must not change the test suite; the extension
# tests switch discovery on again in their own fixture.
os.environ.setdefault('WONDERS_NO_EXTENSIONS', '1')
