#!/usr/bin/env bash
# Start command of the RunPod pods that serve as GitHub Actions runners for
# .github/workflows/test-gpu.yml. `ci.sh launch` sends this file, compressed
# and base64-encoded, as the pod's start command, with these variables:
#   JIT_CONFIG       single-use runner registration (GitHub JIT runner config)
#   RUNNER_VERSION   actions/runner release to install
#   CI_POD_DEADLINE  Unix time by which the pod must be gone
# RunPod itself sets RUNPOD_POD_ID and RUNPOD_API_KEY, a key that can only
# manage this pod.
#
# A pod is billed until it is terminated, so this script makes sure it does
# not outlive its job:
#   - it terminates the pod as soon as the runner exits (job done or
#     cancelled, runner failed to start) or when setup fails;
#   - a watchdog terminates the pod at CI_POD_DEADLINE, whatever the runner
#     is doing;
#   - RunPod restarts a container whose command exits; after a restart the
#     runner is not started again, the pod terminates itself instead.

set -uo pipefail

STATE_DIR=${CI_POD_STATE_DIR:-/opt/ci-pod}
RUNNER_DIR=${CI_POD_RUNNER_DIR:-/opt/actions-runner}
RUNPOD_GRAPHQL_URL=${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}

log() { printf '[ci-pod %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

apt_install() {
  local i
  for i in 1 2 3 4 5; do
    apt-get update -qq &&
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@" &&
      return 0
    log "apt-get install $* failed (attempt $i/5)"
    sleep $((i * 10))
  done
  return 1
}

# Asks RunPod to terminate this pod, or to stop it if termination is refused
# (a stopped pod no longer pays for its GPU; the workflow's cleanup and the
# reaper terminate it later). Keeps trying until the pod is killed: never
# returns.
terminate_self() {
  trap - EXIT
  log "Terminating this pod: $1"
  local op doc resp
  while :; do
    if [[ -z ${RUNPOD_POD_ID:-} || -z ${RUNPOD_API_KEY:-} ]]; then
      log "RUNPOD_POD_ID or RUNPOD_API_KEY is not set, cannot terminate this pod;" \
        "the workflow's cleanup job or the reaper will."
    else
      command -v curl >/dev/null || apt_install curl ca-certificates
      for op in podTerminate podStop; do
        # shellcheck disable=SC2016 # $input is a GraphQL variable
        if [[ $op == podTerminate ]]; then
          doc='mutation($input: PodTerminateInput!) { podTerminate(input: $input) }'
        else
          doc='mutation($input: PodStopInput!) { podStop(input: $input) { id } }'
        fi
        resp=$(curl -sS --max-time 30 -X POST "$RUNPOD_GRAPHQL_URL" \
          -H 'Content-Type: application/json' \
          -H "Authorization: Bearer $RUNPOD_API_KEY" \
          --data-binary @- 2>&1 <<EOF
{"query": "$doc", "variables": {"input": {"podId": "$RUNPOD_POD_ID"}}}
EOF
        )
        if [[ $resp == *'"data"'* && $resp != *'"errors"'* ]]; then
          log "RunPod accepted $op, waiting to be killed."
          sleep 300
          continue 2
        fi
        log "RunPod refused $op: ${resp:0:300}"
      done
    fi
    sleep 60
  done
}

start_watchdog() {
  local deadline=${CI_POD_DEADLINE:-}
  if ! [[ $deadline =~ ^[0-9]+$ ]]; then
    deadline=$(($(date +%s) + 3600))
    log "CI_POD_DEADLINE is not set, using now + 60 min."
  fi
  log "Watchdog: this pod terminates itself at $(date -u -d "@$deadline" +%FT%TZ) at the latest."
  (
    trap - EXIT
    now=$(date +%s)
    if ((deadline > now)); then sleep $((deadline - now)); fi
    terminate_self "time limit (CI_POD_DEADLINE) reached"
  ) &
}

main() {
  mkdir -p "$STATE_DIR" || exit 1
  if [[ -e $STATE_DIR/started ]]; then
    terminate_self "the container restarted after the runner had been started"
  fi
  : >"$STATE_DIR/started"

  # From here on, every way out of this script terminates the pod.
  trap 'terminate_self "start script exited (status $?)"' EXIT
  start_watchdog

  if [[ -z ${JIT_CONFIG:-} || -z ${RUNNER_VERSION:-} ]]; then
    log "JIT_CONFIG or RUNNER_VERSION is not set."
    exit 1
  fi

  log "Installing the runner's dependencies."
  apt_install curl ca-certificates tar git libicu-dev || exit 1
  id -u runner >/dev/null 2>&1 || useradd -m -s /bin/bash runner || exit 1

  log "Installing actions/runner $RUNNER_VERSION."
  mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR" || exit 1
  curl -fsSL --retry 5 --retry-delay 5 -o runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" ||
    exit 1
  tar xzf runner.tar.gz && rm -f runner.tar.gz && chown -R runner:runner "$RUNNER_DIR" || exit 1

  # A JIT runner runs a single job, then exits.
  log "Starting the runner."
  su - runner -c "cd '$RUNNER_DIR' && ./run.sh --jitconfig '$JIT_CONFIG'"
  log "Runner exited with status $?."
}

main "$@"
