#!/bin/bash
# Profile the output writer tests on a CI agent, first through ParallelTestRunner exactly
# like the "all tests" step, then as standalone processes, while sampling the host load and
# recording where the tests write their files. Everything is collected into one tarball
# uploaded as a Buildkite artifact.
#
# The artifact is public: record process names rather than command lines, a whitelist of
# environment variables, and mount points without their sources.
set -uo pipefail

TESTS=(simulation/netcdf_writer simulation/zarr_writer lagrangian_particles/tracking)

export CUDA_VISIBLE_DEVICES="-1"
export TEST_ARCHITECTURE="CPU"
export JULIA_NUM_GC_THREADS="4,1"

# On Nautilus we have old GPUs which require CUDA toolkit v12, can't use v13.
echo -e '[CUDA_Runtime_jll]\nversion = "12.9"' > test/LocalPreferences.toml

ROOT="$PWD"
OUT="$ROOT/instrumentation"
rm -rf "$OUT"
mkdir -p "$OUT/ptr" "$OUT/standalone"

CGROUP_DIR="/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
ENV_WHITELIST=(JULIA_VERSION JULIA_NUM_THREADS JULIA_NUM_GC_THREADS JULIA_CPU_THREADS JULIA_DEPOT_PATH
               JULIA_PKG_SERVER_REGISTRY_PREFERENCE JULIA_CUDA_USE_COMPAT OPENBLAS_NUM_THREADS
               OMP_NUM_THREADS MKL_NUM_THREADS TMPDIR TMP TEMP HOME CUDA_VISIBLE_DEVICES TEST_ARCHITECTURE)

# Local, temporary and network file systems a test could write to.
MOUNT_TYPES=ext2,ext3,ext4,xfs,btrfs,zfs,tmpfs,overlay,nfs,nfs4,lustre,gpfs,beegfs,cifs,fuse.sshfs

filesystem_info() {
    echo "### tempdir candidates"
    for d in "$ROOT" /tmp "${TMPDIR:-/tmp}" /dev/shm /var/tmp "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" /scratch /local /localscratch /tmp_local; do
        [ -d "$d" ] || continue
        echo "$d: writable=$([ -w "$d" ] && echo yes || echo no)"
        findmnt -T "$d" -o TARGET,FSTYPE,SIZE,AVAIL,USE% 2>&1 | tail -n +2
    done
    echo "### mounts"
    findmnt -rn -t "$MOUNT_TYPES" -o TARGET,FSTYPE,SIZE,AVAIL,USE% 2>&1 | grep -vE '^/(run/user|var/lib/docker|snap)/'
    echo "### block devices"
    lsblk -e 7 -o NAME,TYPE,SIZE,ROTA,TRAN,MODEL,MOUNTPOINTS 2>&1
}

system_info() {
    echo "### date";            date -u
    echo "### uptime";          uptime
    echo "### nproc";           nproc
    echo "### lscpu";           lscpu
    echo "### free";            free -m
    for f in cpu.max cpu.weight cpu.stat cpuset.cpus.effective memory.max memory.high memory.current memory.stat io.stat; do
        echo "### cgroup $f"; cat "$CGROUP_DIR/$f" 2>&1
    done
    echo "### pressure";        for f in /proc/pressure/*; do echo "$f: $(tr '\n' ' ' < "$f")"; done
    echo "### affinity";        taskset -pc $$
    echo "### ulimit";          ulimit -a
    echo "### env"
    for v in "${ENV_WHITELIST[@]}"; do echo "$v=${!v-<unset>}"; done
    filesystem_info
    echo "### busiest processes on the host"
    ps -eo pid,etime,time,pcpu,nlwp,rss,stat,comm --sort=-pcpu | head -40
}

sample_host() {
    while true; do
        echo "### $(date -u +%FT%T)"
        echo "loadavg: $(cat /proc/loadavg)"
        for f in /proc/pressure/*; do echo "$(basename "$f"): $(tr '\n' ' ' < "$f")"; done
        echo "cgroup cpu.stat: $(tr '\n' ' ' < "$CGROUP_DIR/cpu.stat" 2>/dev/null)"
        echo "cgroup memory.current: $(cat "$CGROUP_DIR/memory.current" 2>/dev/null)"
        echo "cgroup io.stat: $(tr '\n' ' ' < "$CGROUP_DIR/io.stat" 2>/dev/null)"
        grep -E '^(procs_running|procs_blocked|ctxt) ' /proc/stat | tr '\n' ' '; echo
        vmstat 1 2 | tail -1
        ps -eo pid,ppid,etime,time,pcpu,nlwp,rss,stat,comm --sort=-pcpu | head -25
        sleep 15
    done
}

descendants() {
    local p
    for p in $(pgrep -P "$1"); do
        echo "$p"
        descendants "$p"
    done
}

# Every 2 seconds, record the files (other than libraries and caches) open in our Julia processes.
sample_open_files() {
    while true; do
        for p in $(descendants $$); do
            [ "$(cat /proc/$p/comm 2>/dev/null)" = julia ] || continue
            for fd in /proc/$p/fd/*; do
                target=$(readlink "$fd" 2>/dev/null) || continue
                case "$target" in
                    /*.so*|/*.ji|/*.cov|/dev/*|/proc/*|/etc/*|"$OUT"/*|*/compiled/*|*/artifacts/*|*/packages/*|*/juliaup/*|*/registries/*) ;;
                    /*) echo "$(date -u +%T) $p $target" ;;
                esac
            done
        done
        sleep 2
    done
}

fs_benchmark() {
    local dirs=("$ROOT" /tmp "${TMPDIR:-/tmp}" /dev/shm /var/tmp "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}")
    # Also try every other local mount we can write to.
    while read -r target; do
        [ -w "$target" ] && dirs+=("$target")
    done < <(findmnt -rn -t ext4,xfs,btrfs,tmpfs -o TARGET | grep -vE '^/(proc|sys|dev|run|boot|snap)')
    julia +$JULIA_VERSION --startup-file=no --project="$ROOT/test" "$ROOT/.buildkite/filesystem_benchmark.jl" \
        $(printf '%s\n' "${dirs[@]}" | awk '!seen[$0]++')
}

system_info > "$OUT/system_before.txt" 2>&1

# The standalone runs (phase 2) and the filesystem benchmark need the test environment.
JULIA_FLAGS=(-O0 --check-bounds=yes --depwarn=yes --startup-file=no --color=no "--code-coverage=@$ROOT" "--project=$ROOT/test")

julia +$JULIA_VERSION "${JULIA_FLAGS[@]}" -e \
    'using Pkg; Pkg.instantiate(); using Oceananigans, NCDatasets, Zarr, JLD2, CUDA, Profile, Serialization' \
    > "$OUT/standalone/precompile.txt" 2>&1

# I/O latency of the candidate output directories before our tests load the host, to compare with
# the same measurement at the end.
fs_benchmark > "$OUT/filesystem_benchmark_before.txt" 2>&1
cat "$OUT/filesystem_benchmark_before.txt"

sample_host > "$OUT/host_samples.txt" 2>&1 &
SAMPLER_PID=$!
sample_open_files > "$OUT/open_files_raw.txt" 2>&1 &
OPEN_FILES_PID=$!

##### Phase 1: ParallelTestRunner, with the same command as the "all tests" step.

touch "$OUT/phase1.marker"
phase1_start=$(date +%s)
OCEANANIGANS_PROFILE_DIR="$OUT/ptr" julia +$JULIA_VERSION -O0 --color=yes --project -e \
    'using Pkg;
     Pkg.test("Oceananigans";
       coverage=true,
       test_args=`--verbose '"${TESTS[*]}"'`)
    ' 2>&1 | tee "$OUT/ptr/log.txt"
test_status=${PIPESTATUS[0]}
echo "phase 1 (ParallelTestRunner) wall time: $(( $(date +%s) - phase1_start )) s" | tee -a "$OUT/phases.txt"

# Output files the tests left behind, and where they live.
{
    for d in "$ROOT" /tmp "${TMPDIR:-/tmp}"; do
        echo "### files newer than phase 1 start under $d ($(findmnt -T "$d" -no TARGET,FSTYPE))"
        find "$d" -xdev -newer "$OUT/phase1.marker" -type f \( -name '*.nc' -o -name '*.jld2' -o -path '*.zarr/*' \) \
            -user "$(id -u)" -printf '%s %p\n' 2>/dev/null | grep -v "^[0-9]* $OUT/" | head -500
    done
} > "$OUT/leftover_output_files.txt" 2>&1

##### Phase 2: the same files in standalone processes, concurrently, with the flags Pkg.test passes.


phase2_start=$(date +%s)
pids=()
for name in "${TESTS[@]}"; do
    tag=${name//\//__}
    mkdir -p "$OUT/standalone-cwd/$tag"
    (cd "$OUT/standalone-cwd/$tag" && \
     julia +$JULIA_VERSION "${JULIA_FLAGS[@]}" \
         "$ROOT/.buildkite/profile_test_file.jl" "$name" "$OUT/standalone/$tag" \
         > "$OUT/standalone/$tag.log.txt" 2>&1) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "phase 2 (standalone) wall time: $(( $(date +%s) - phase2_start )) s" | tee -a "$OUT/phases.txt"

kill "$SAMPLER_PID" "$OPEN_FILES_PID"

##### Phase 3: I/O latency of the candidate output directories again, right after our tests.

fs_benchmark > "$OUT/filesystem_benchmark_after.txt" 2>&1
cat "$OUT/filesystem_benchmark_after.txt"

# Unique open files with the mount they belong to.
awk '{print $3}' "$OUT/open_files_raw.txt" | sort | uniq -c | sort -rn > "$OUT/open_files_counts.txt"
{
    awk '{print $3}' "$OUT/open_files_raw.txt" | xargs -r -n1 dirname | sort -u | while read -r d; do
        existing=$d
        while [ ! -e "$existing" ]; do existing=$(dirname "$existing"); done
        echo "$d -> $(findmnt -T "$existing" -no TARGET,FSTYPE)"
    done
} > "$OUT/open_files_mounts.txt"

system_info > "$OUT/system_after.txt" 2>&1

tarball="instrumentation-${BUILDKITE_BUILD_NUMBER}-${BUILDKITE_JOB_ID}.tgz"
tar -czf "$tarball" --exclude=standalone-cwd -C "$ROOT" instrumentation
ls -lh "$tarball"
buildkite-agent artifact upload "$tarball"

exit "$test_status"
