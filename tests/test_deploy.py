"""The CI/CD pipeline and the VM's continuous-deployment files."""

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())


def test_pipeline_tests_then_builds_then_verifies_the_deploy():
    jobs = WORKFLOW["jobs"]
    steps = " ".join(str(s.get("run", "")) for s in jobs["test"]["steps"])
    assert "ruff check" in steps and "pytest" in steps
    assert jobs["build"]["needs"] == "test" and "refs/heads/main" in jobs["build"]["if"]
    assert jobs["deploy"]["needs"] == "build" and jobs["deploy"]["environment"]["name"] == "production"
    assert WORKFLOW["permissions"] == {"contents": "read"}  # write access only where a job needs it


def test_build_signs_in_without_a_stored_key_and_stamps_the_version():
    build = WORKFLOW["jobs"]["build"]
    assert build["permissions"]["id-token"] == "write"
    auth = next(s for s in build["steps"] if str(s.get("uses", "")).startswith("google-github-actions/auth"))
    assert set(auth["with"]) == {"workload_identity_provider", "service_account"}  # no credentials_json
    push = next(s for s in build["steps"] if str(s.get("uses", "")).startswith("docker/build-push-action"))
    assert "APP_VERSION=${{ github.sha }}" in push["with"]["build-args"]


def test_vm_autodeploy_rolls_back_and_never_touches_the_env_file():
    script = (ROOT / "deploy" / "vm" / "autodeploy.sh").read_text()
    assert "rolling back" in script and "git reset --quiet --hard origin/main" in script
    assert '["services"]["web"]["image"]' in script  # the web image by name, never by list position
    assert "updated the deploy script" in script  # fixes to this script reach the server by themselves
    commands = "\n".join(line.split("#")[0] for line in script.splitlines())  # code, not comments
    assert not re.search(r"(\brm\b|\bcp\b|\bmv\b|>)[^\n]*\.env", commands)  # never writes or deletes .env
    for name in ("autodeploy.sh", "install.sh"):
        assert subprocess.run(["sh", "-n", str(ROOT / "deploy" / "vm" / name)]).returncode == 0
    timer = (ROOT / "deploy" / "vm" / "handbook-autodeploy.timer").read_text()
    assert "OnUnitActiveSec=1min" in timer


def test_compose_runs_the_registry_image_on_the_server():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert compose["services"]["web"]["image"].startswith("${WEB_IMAGE")
    assert "https" in compose["services"]["caddy"]["profiles"]
