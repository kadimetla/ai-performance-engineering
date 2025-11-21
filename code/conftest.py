"""Global pytest configuration.

We explicitly disable auto-loading of external pytest plugins to prevent
environment-provided plugins from interfering with test discovery and capture
in this repository's harness.
"""

import os

# Guard against site-wide plugins that can change stdout handling (causing
# Illegal seek/OSError on teardown in CI and local shells).
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
