#!/bin/sh
# One-time VM setup for continuous deployment (run with sudo from the repository checkout):
#     sudo sh deploy/vm/install.sh
# Needs: Docker, the checkout at /home/Rafay-Dev/handbook-rag-assistant with its .env, and the VM's
# service account allowed to read Artifact Registry (gcloud auth configure-docker for root).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1
install -m 755 "$HERE/autodeploy.sh" /usr/local/bin/handbook-autodeploy  # outside the checkout it updates
install -m 644 "$HERE/handbook-autodeploy.service" /etc/systemd/system/handbook-autodeploy.service
install -m 644 "$HERE/handbook-autodeploy.timer" /etc/systemd/system/handbook-autodeploy.timer
systemctl daemon-reload
systemctl enable --now handbook-autodeploy.timer
echo "continuous deployment on: journalctl -u handbook-autodeploy -f"
