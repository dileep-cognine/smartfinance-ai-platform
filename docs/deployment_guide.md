# Deployment Guide

Follow the root README for local installation. Production deployments must
replace placeholder secrets, terminate TLS at an authenticated gateway, remove
host port mappings that are not operationally required, use managed encrypted
volumes, configure backups, restrict outbound provider access, and forward
structured logs and alerts to the approved observability platform.
