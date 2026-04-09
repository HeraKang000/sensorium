#!/bin/bash
# If GH_TOKEN is set (private repo), pass it as Authorization header.
# For a public repo, no token needed — just works.
if [ -n "${GH_TOKEN:-}" ]; then
    curl -fsSL \
        -H "Authorization: token ${GH_TOKEN}" \
        "https://raw.githubusercontent.com/HeraKang000/sensorium/main/provisioning.sh" | bash
else
    curl -fsSL \
        "https://raw.githubusercontent.com/HeraKang000/sensorium/main/provisioning.sh" | bash
fi
