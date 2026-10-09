"""Test-wide settings."""

import os
import tempfile

# Installed extensions (~/.wonders/extensions, plugins) must not change the test suite; the extension
# tests switch discovery on again in their own fixture.
os.environ.setdefault('WONDERS_NO_EXTENSIONS', '1')

# Saved games, ratings and extensions live under WONDERS_HOME: keep the suite away from ~/.wonders.
os.environ['WONDERS_HOME'] = tempfile.mkdtemp(prefix='wonders-test-home-')
