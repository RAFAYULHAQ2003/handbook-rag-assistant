"""Settings for the test suite: runs without a .env file, with a throwaway key."""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-insecure-key")
os.environ.setdefault("REFERENCE_DATE", "")

from webapp.settings import *  # noqa: E402,F403
