"""Django settings refuse unsafe production defaults."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_settings(**env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("DJANGO_", "COMPANY_"))}
    # explicit empty values: python-dotenv never overrides existing variables, so a local .env cannot leak in
    clean.update({"DJANGO_DEBUG": "", "DJANGO_SECRET_KEY": "", "COMPANY_TIMEZONE": ""}, **env)
    clean["DJANGO_SETTINGS_MODULE"] = "webapp.settings"
    code = "from django.conf import settings; print(settings.DEBUG, settings.TIME_ZONE)"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean, capture_output=True, text=True)


def test_debug_is_off_by_default_and_needs_a_secret_key():
    result = load_settings()
    assert result.returncode != 0
    assert "DJANGO_SECRET_KEY" in result.stderr


def test_production_with_a_key_runs_without_debug():
    result = load_settings(DJANGO_SECRET_KEY="a-real-secret")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "America/Chicago"]


def test_local_debug_allows_the_dev_key():
    result = load_settings(DJANGO_DEBUG="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == "True"
