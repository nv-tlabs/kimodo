#!/usr/bin/env bash
set -euo pipefail

HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"
HOST_USER="${HOST_USER:-user}"

if [[ -z "${HOST_UID}" || -z "${HOST_GID}" ]]; then
  if [[ -d /workspace ]]; then
    HOST_UID="$(stat -c %u /workspace)"
    HOST_GID="$(stat -c %g /workspace)"
  else
    HOST_UID="${HOST_UID:-1000}"
    HOST_GID="${HOST_GID:-1000}"
  fi
fi

# If HOST_USER == "root" or HOST_UID == 0, we're being asked to run as the
# image's existing root user — don't try to groupadd/useradd a duplicate.
# This avoids `groupadd: group 'root' already exists` when callers invoke
# `docker compose up` from a root shell (where ${USER} expands to "root"
# in the compose env block) without remembering to override HOST_USER.
if [[ "${HOST_USER}" == "root" ]] || [[ "${HOST_UID}" == "0" ]]; then
  exec "$@"
fi

if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
  groupadd -g "${HOST_GID}" "${HOST_USER}"
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"
fi

exec gosu "${HOST_UID}:${HOST_GID}" "$@"
