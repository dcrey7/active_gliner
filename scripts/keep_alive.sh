#!/usr/bin/env bash
set -uo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 scripts/serve_teacher_gemma.sh" >&2
  exit 2
fi

child=
stop() {
  trap '' INT TERM
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit 0
}
trap stop INT TERM

while true; do
  "$@" &
  child=$!
  wait "$child"
  code=$?
  child=
  printf '%s launch exited with code %s; restarting in 10 s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$code" >&2
  sleep 10 &
  child=$!
  wait "$child"
  child=
done
