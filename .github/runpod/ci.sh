#!/usr/bin/env bash
# RunPod helpers for .github/workflows/test-gpu.yml and gpu-pod-reaper.yml.
#
#   ci.sh launch       create one pod per Python x CUDA combo, each running a
#                      single-use GitHub runner; writes the test matrix to
#                      $GITHUB_OUTPUT
#   ci.sh wait-online  wait until the runners of the pods are online
#   ci.sh cleanup      terminate this run's pods, deregister its runners
#   ci.sh reap         terminate CI pods older than MAX_POD_AGE_MINUTES
#
# A pod and its runner share a name starting with POD_PREFIX
# (gha-instagraal-<run id>-<run attempt>-). Cleanup finds pods by that name,
# so it also terminates pods whose launch step died before reporting them.
#
# Environment (secrets come from the workflow):
#   RUNPOD_API_KEY                           all commands
#   GH_PAT        repo administration token  launch, wait-online, cleanup
#   POD_PREFIX                               launch, wait-online, cleanup
#   COMBOS, GPU_TYPES, POD_TTL_MINUTES       launch
#   MATRIX, WAIT_MINUTES                     wait-online
#   MAX_POD_AGE_MINUTES                      reap

set -euo pipefail

RUNPOD_GRAPHQL_URL=${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}
GITHUB_API_URL=${GITHUB_API_URL:-https://api.github.com}
# Every CI pod name starts with this. The reaper only ever touches these pods,
# never the other pods of the RunPod account.
CI_POD_BASE_PREFIX=gha-instagraal-
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

log() { echo "$*" >&2; }
die() {
  echo "::error::$*" >&2
  exit 1
}
require() {
  local v
  for v; do [[ -n ${!v:-} ]] || die "$v is not set"; done
}

# --- APIs ------------------------------------------------------------------

# Responses go through a file: when curl retries, it truncates its output file,
# whereas on stdout the bodies of the failed attempts would come first.

# runpod_call [--retry] JSON: POST a GraphQL request, print the JSON response.
# Only use --retry for requests that are safe to repeat.
runpod_call() {
  local retry=() out rc=0 body
  if [[ $1 == --retry ]]; then
    retry=(--retry 4 --retry-delay 5 --retry-max-time 120)
    shift
  fi
  out=$(mktemp)
  curl -sS --max-time 60 "${retry[@]}" -o "$out" -X POST "$RUNPOD_GRAPHQL_URL" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    --data-binary @- <<<"$1" || rc=$?
  body=$(<"$out")
  rm -f "$out"
  if ((rc != 0)) || ! jq -e 'type == "object"' >/dev/null 2>&1 <<<"$body"; then
    log "RunPod request failed (curl exit $rc): ${body:0:300}"
    return 1
  fi
  printf '%s\n' "$body"
}

# gh_call [--retry] METHOD PATH [JSON]: call the GitHub REST API as GH_PAT.
# Prints the response body, also on HTTP errors (then returns non-zero).
gh_call() {
  local retry=()
  if [[ $1 == --retry ]]; then
    retry=(--retry 4 --retry-delay 5 --retry-max-time 120)
    shift
  fi
  local method=$1 path=$2 data=${3:-} input=() out rc=0
  [[ -z $data ]] || input=(--data-binary @-)
  out=$(mktemp)
  curl -sS --fail-with-body --max-time 30 "${retry[@]}" -o "$out" -X "$method" \
    -H "Authorization: Bearer $GH_PAT" \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2022-11-28' \
    "${input[@]}" "$GITHUB_API_URL/$path" <<<"$data" || rc=$?
  cat "$out"
  rm -f "$out"
  return "$rc"
}

# list_pods PREFIX: one "id<TAB>name<TAB>desiredStatus<TAB>age in minutes"
# line per pod whose name starts with PREFIX and that is not terminated yet.
# Age is -1 when RunPod does not report the creation time.
list_pods() {
  local resp
  resp=$(runpod_call --retry "$(jq -nc '{query: "query { myself { pods { id name desiredStatus createdAt } } }"}')") ||
    return 1
  if ! jq -e '.data.myself.pods | type == "array"' >/dev/null <<<"$resp"; then
    log "Unexpected RunPod response: ${resp:0:300}"
    return 1
  fi
  jq -r --arg prefix "$1" --argjson now "$(date +%s)" '
    .data.myself.pods[]
    | select(((.name // "") | startswith($prefix)) and .desiredStatus != "TERMINATED")
    | (try (.createdAt | sub("\\.[0-9]+"; "") | fromdateiso8601) catch null) as $created
    | [.id, .name, .desiredStatus, (if $created then ($now - $created) / 60 | floor else -1 end)]
    | @tsv' <<<"$resp"
}

terminate_pod() {
  local resp
  resp=$(runpod_call --retry "$(jq -nc --arg id "$1" '{
    query: "mutation($input: PodTerminateInput!) { podTerminate(input: $input) }",
    variables: {input: {podId: $id}}}')") || return 1
  if jq -e '.errors | length > 0' >/dev/null <<<"$resp"; then
    log "  podTerminate $1: $(jq -c '[.errors[].message]' <<<"$resp")"
    return 1
  fi
}

# terminate_pods PODS PREFIX MIN_AGE: terminate PODS (lines of list_pods),
# then list the pods named PREFIX* at least MIN_AGE minutes old (any age if
# MIN_AGE is 0) and terminate those again, until none is left. Fails if some
# survive five rounds.
terminate_pods() {
  local pods=$1 prefix=$2 min_age=$3 round listed id name status age
  [[ -n $pods ]] || return 0
  for round in 1 2 3 4 5; do
    while IFS=$'\t' read -r id name status age; do
      log "Terminating pod $name ($id, $status, $age min old)"
      terminate_pod "$id" || true
    done <<<"$pods"
    sleep $((round * 5))
    # If listing fails, the same pods are terminated again.
    if listed=$(list_pods "$prefix"); then
      pods=$(awk -F'\t' -v min="$min_age" 'min == 0 || $4 >= min' <<<"$listed")
      [[ -n $pods ]] || return 0
    fi
  done
  while IFS=$'\t' read -r id name status age; do
    echo "::error::Pod $name ($id, $status) may still be running: check https://www.runpod.io/console/pods" >&2
  done <<<"$pods"
  return 1
}

# list_runners PREFIX: one "id<TAB>name<TAB>status" line per runner of the
# repository whose name starts with PREFIX.
list_runners() {
  local page=1 body
  while :; do
    body=$(gh_call --retry GET "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100&page=$page") || return 1
    jq -r --arg prefix "$1" '.runners[] | select(.name | startswith($prefix)) | [.id, .name, .status] | @tsv' <<<"$body"
    (($(jq '.runners | length' <<<"$body") == 100)) || return 0
    page=$((page + 1))
  done
}

# --- launch ----------------------------------------------------------------

latest_runner_version() {
  local auth=()
  [[ -z ${GITHUB_TOKEN:-} ]] || auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
  curl -fsSL --retry 4 --max-time 30 "${auth[@]}" -H 'Accept: application/vnd.github+json' \
    "$GITHUB_API_URL/repos/actions/runner/releases/latest" | jq -re '.tag_name | ltrimstr("v")'
}

# create_pod NAME IMAGE CUDA_VERSION GPU_TYPE JIT_CONFIG: print the pod id.
create_pod() {
  local payload resp id
  payload=$(jq -nc --arg name "$1" --arg image "$2" --arg min_cuda "$(cut -d. -f1,2 <<<"$3")" \
    --arg gpu "$4" --arg jit "$5" --arg cmd "$DOCKER_ARGS" --arg runner_version "$RUNNER_VERSION" \
    --arg deadline "$DEADLINE" --arg deadline_iso "$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)" '{
      query: "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id } }",
      variables: {input: {
        name: $name,
        imageName: $image,
        gpuTypeId: $gpu,
        gpuCount: 1,
        cloudType: "SECURE",
        # Only hosts whose driver supports the image CUDA version.
        minCudaVersion: $min_cuda,
        containerDiskInGb: 20,
        volumeInGb: 0,
        volumeMountPath: "/workspace",
        dockerArgs: $cmd,
        # Accepted but not enforced by RunPod (Nov 2026); pod-startup.sh
        # enforces CI_POD_DEADLINE. Kept in case RunPod starts honouring it.
        terminateAfter: $deadline_iso,
        env: [
          {key: "JIT_CONFIG", value: $jit},
          {key: "RUNNER_VERSION", value: $runner_version},
          {key: "CI_POD_DEADLINE", value: $deadline}
        ]
      }}}')
  # Not retried by curl: a repeated request could create a second pod.
  resp=$(runpod_call "$payload") || return 1
  id=$(jq -r '.data.podFindAndDeployOnDemand.id // empty' <<<"$resp")
  if [[ -z $id ]]; then
    log "    $(jq -c '[.errors[]?.message]' <<<"$resp")"
    return 1
  fi
  echo "$id"
}

# create_pod_with_retries NAME IMAGE CUDA_VERSION JIT_CONFIG: try each GPU type
# in turn, three rounds. Prints "pod_id<TAB>gpu".
create_pod_with_retries() {
  local round gpu id
  for round in 1 2 3; do
    for gpu in "${GPU_TYPE_LIST[@]}"; do
      if id=$(create_pod "$1" "$2" "$3" "$gpu" "$4"); then
        printf '%s\t%s\n' "$id" "$gpu"
        return 0
      fi
      log "  $1: no pod on $gpu (round $round/3)"
    done
    ((round == 3)) || sleep $((round * 30))
  done
  return 1
}

cmd_launch() {
  require RUNPOD_API_KEY GH_PAT GITHUB_REPOSITORY GITHUB_OUTPUT POD_PREFIX COMBOS GPU_TYPES POD_TTL_MINUTES
  [[ $POD_PREFIX == "$CI_POD_BASE_PREFIX"?* ]] || die "POD_PREFIX must start with $CI_POD_BASE_PREFIX"
  [[ $POD_TTL_MINUTES =~ ^[1-9][0-9]*$ ]] || die "POD_TTL_MINUTES must be a number of minutes"
  mapfile -t GPU_TYPE_LIST < <(sed 's/^[[:space:]]*//; s/[[:space:]]*$//; /^$/d' <<<"$GPU_TYPES")
  ((${#GPU_TYPE_LIST[@]} > 0)) || die "GPU_TYPES is empty"
  jq -e 'type == "array" and length > 0 and all(.[]; (.["python-version"] | type) == "string"
    and (.["cuda-version"] | type) == "string" and (.image | type) == "string")' >/dev/null <<<"$COMBOS" ||
    die "COMBOS must be a JSON array of {python-version, cuda-version, image}"

  RUNNER_VERSION=$(latest_runner_version) || die "Could not find the latest actions/runner release"
  DEADLINE=$(($(date +%s) + POD_TTL_MINUTES * 60))
  # The pod runs pod-startup.sh. Shipping it compressed and base64-encoded in
  # the start command avoids quoting issues and needs no access to the repo.
  DOCKER_ARGS="bash -c 'echo $(gzip -9cn "$SCRIPT_DIR/pod-startup.sh" | base64 -w0) | base64 -d | gzip -dc > /ci-pod-startup.sh && exec bash /ci-pod-startup.sh'"
  log "Pods terminate themselves at $(date -u -d "@$DEADLINE" +%FT%TZ) at the latest (POD_TTL_MINUTES=$POD_TTL_MINUTES)."

  local include='[]' summary="" row py cuda image label body jit pod
  while IFS= read -r row; do
    py=$(jq -r '.["python-version"]' <<<"$row")
    cuda=$(jq -r '.["cuda-version"]' <<<"$row")
    image=$(jq -r '.image' <<<"$row")
    label="${POD_PREFIX}py${py//./}-cuda${cuda//./}"

    # A JIT runner config registers one runner, for one job.
    body=$(jq -nc --arg name "$label" '{name: $name, runner_group_id: 1, labels: [$name], work_folder: "_work"}')
    body=$(gh_call POST "repos/$GITHUB_REPOSITORY/actions/runners/generate-jitconfig" "$body") ||
      die "Could not create a runner for $label: ${body:0:300}"
    jit=$(jq -r '.encoded_jit_config // empty' <<<"$body")
    [[ -n $jit ]] || die "No runner config returned for $label"
    echo "::add-mask::$jit"

    # On failure, the pods created so far are terminated by the cleanup job.
    pod=$(create_pod_with_retries "$label" "$image" "$cuda" "$jit") || die "Could not create a pod for $label"
    log "✓ $label → pod ${pod%%$'\t'*} on ${pod#*$'\t'} (py$py + cuda$cuda)"
    summary+="| \`$label\` | \`${pod%%$'\t'*}\` | ${pod#*$'\t'} |"$'\n'
    include=$(jq -c --arg label "$label" --arg py "$py" --arg cuda "$cuda" \
      '. + [{label: $label, "python-version": $py, "cuda-version": $cuda}]' <<<"$include")
  done < <(jq -c '.[]' <<<"$COMBOS")

  echo "matrix=$(jq -c '{include: .}' <<<"$include")" >>"$GITHUB_OUTPUT"
  if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
    printf '### GPU pods\n\nTerminated by %s at the latest.\n\n| Runner | Pod | GPU |\n|---|---|---|\n%s' \
      "$(date -u -d "@$DEADLINE" +%FT%TZ)" "$summary" >>"$GITHUB_STEP_SUMMARY"
  fi
}

# --- wait-online -----------------------------------------------------------

cmd_wait_online() {
  require RUNPOD_API_KEY GH_PAT GITHUB_REPOSITORY POD_PREFIX MATRIX WAIT_MINUTES
  local labels label runners pods pods_ok status pending deadline now
  local -A missing_since=()
  mapfile -t labels < <(jq -r '.include[].label' <<<"$MATRIX")
  ((${#labels[@]} > 0)) || die "MATRIX has no runner"
  deadline=$(($(date +%s) + WAIT_MINUTES * 60))
  while :; do
    runners=$(list_runners "$POD_PREFIX") || runners=""
    pods_ok=1
    pods=$(list_pods "$POD_PREFIX") || pods_ok=0
    now=$(date +%s)
    pending=()
    for label in "${labels[@]}"; do
      status=$(awk -F'\t' -v n="$label" '$2 == n {print $3}' <<<"$runners")
      [[ $status != online ]] || continue
      pending+=("$label (${status:-not registered})")
      # A pod terminates itself when its runner fails to start: stop waiting
      # once it has been missing or stopped for a minute (the margin covers
      # a pod list that lags behind pod creation).
      ((pods_ok)) || continue
      if awk -F'\t' -v n="$label" '$2 == n && $3 != "EXITED" && $3 != "DEAD" {f = 1} END {exit !f}' <<<"$pods"; then
        unset 'missing_since[$label]'
      else
        missing_since[$label]=${missing_since[$label]:-$now}
        ((now - missing_since[$label] < 60)) ||
          die "The pod of $label is gone: its runner failed to start (see the pod log in the RunPod console)."
      fi
    done
    if ((${#pending[@]} == 0)); then
      log "All ${#labels[@]} runners are online."
      return 0
    fi
    (($(date +%s) < deadline)) || die "Runners not online after $WAIT_MINUTES min: ${pending[*]}"
    log "Waiting for ${#pending[@]} runner(s): ${pending[*]}"
    sleep 15
  done
}

# --- cleanup ---------------------------------------------------------------

cmd_cleanup() {
  require RUNPOD_API_KEY POD_PREFIX
  [[ $POD_PREFIX == "$CI_POD_BASE_PREFIX"?* ]] || die "POD_PREFIX must start with $CI_POD_BASE_PREFIX"
  local pods failed=0 runners id name status
  if pods=$(list_pods "$POD_PREFIX"); then
    [[ -n $pods ]] || log "No pod named $POD_PREFIX* is left."
    terminate_pods "$pods" "$POD_PREFIX" 0 || failed=1
  else
    echo "::error::Could not list RunPod pods: check https://www.runpod.io/console/pods for pods named $POD_PREFIX*" >&2
    failed=1
  fi

  if [[ -z ${GH_PAT:-} || -z ${GITHUB_REPOSITORY:-} ]]; then
    log "GH_PAT or GITHUB_REPOSITORY not set: runners not deregistered."
  elif runners=$(list_runners "$POD_PREFIX"); then
    while IFS=$'\t' read -r id name status; do
      [[ -n $id ]] || continue
      if gh_call --retry DELETE "repos/$GITHUB_REPOSITORY/actions/runners/$id" >/dev/null; then
        log "Deregistered runner $name ($status)"
      else
        echo "::warning::Could not deregister runner $name ($id)" >&2
      fi
    done <<<"$runners"
  else
    echo "::warning::Could not list the repository's runners" >&2
  fi
  return "$failed"
}

# --- reap ------------------------------------------------------------------

cmd_reap() {
  require RUNPOD_API_KEY MAX_POD_AGE_MINUTES
  [[ $MAX_POD_AGE_MINUTES =~ ^[1-9][0-9]*$ ]] || die "MAX_POD_AGE_MINUTES must be a number of minutes"
  local pods stale unknown failed=0
  pods=$(list_pods "$CI_POD_BASE_PREFIX") || die "Could not list RunPod pods"
  unknown=$(awk -F'\t' '$4 < 0' <<<"$pods")
  if [[ -n $unknown ]]; then
    echo "::error::RunPod did not report the creation time of: $(cut -f2 <<<"$unknown" | paste -sd' ')" >&2
    failed=1
  fi
  stale=$(awk -F'\t' -v max="$MAX_POD_AGE_MINUTES" '$4 >= max' <<<"$pods")
  log "$(grep -c . <<<"$pods") CI pod(s), $(grep -c . <<<"$stale") older than $MAX_POD_AGE_MINUTES min."
  if [[ -n $stale ]]; then
    terminate_pods "$stale" "$CI_POD_BASE_PREFIX" "$MAX_POD_AGE_MINUTES" || true
    # Fail even if they are gone now: the other safeguards missed these pods.
    echo "::error::Terminated CI pods older than $MAX_POD_AGE_MINUTES min, which the workflow and the" \
      "pod watchdog should have terminated: $(cut -f2 <<<"$stale" | paste -sd' '). Check the runs that created them." >&2
    failed=1
  fi
  return "$failed"
}

case ${1:-} in
launch) cmd_launch ;;
wait-online) cmd_wait_online ;;
cleanup) cmd_cleanup ;;
reap) cmd_reap ;;
*) die "usage: $0 launch|wait-online|cleanup|reap" ;;
esac
