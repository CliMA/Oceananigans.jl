#!/usr/bin/env bash
#
# Common Buildkite test runner. Sets architecture-specific env vars,
# strips the emoji prefix from the group name, instantiates the requested
# project, runs test/runtests.jl, and uploads coverage as an artifact.
#
# Usage:
#   run_tests.sh <julia_version> <project> <architecture> <group> [extra julia flags...]
#
# Examples:
#   run_tests.sh 1.12.4  test            CPU "🐇 unit"
#   run_tests.sh 1.12.4  test/extensions GPU "🍱 xesmf"
#   run_tests.sh 1.10.10 test/extensions CPU "👹 reactant_1" --check-bounds=auto
set -u

JULIA_VERSION="$1"
PROJECT="$2"
ARCH="$3"
GROUP="$4"
shift 4

cd "$OCEANANIGANS_DIR"

if [[ "$ARCH" == "CPU" ]]; then
  export CUDA_VISIBLE_DEVICES="-1"
else
  export CUDA_VISIBLE_DEVICES="0"
fi
export TEST_ARCHITECTURE="$ARCH"

# Strip leading emoji + space from the group label
export TEST_GROUP="${GROUP#* }"
echo "TEST_GROUP=$TEST_GROUP"

julia +"$JULIA_VERSION" -O0 --color=yes --project="$PROJECT" -e 'using Pkg; Pkg.instantiate()'

# Run tests, but preserve exit status so we can upload coverage even on failure
set +e
julia +"$JULIA_VERSION" -O0 --color=yes --project="$PROJECT" --code-coverage=user "$@" test/runtests.jl
test_status=$?
set -e

echo "PWD: $(pwd)"
echo "Coverage files produced: $(find . -type f -name '*.cov' | wc -l)"
find . -type f -name "*.cov" | head -n 20 || true

group_slug="${GROUP//[^A-Za-z0-9_.-]/_}"
tarball="coverage-${BUILDKITE_BUILD_NUMBER}-${BUILDKITE_JOB_ID}-${ARCH}-${group_slug}.tgz"

find . -type f -name "*.cov" > cov_list.txt
echo "Coverage files found: $(wc -l < cov_list.txt)"

tar -czf "$tarball" -T cov_list.txt
ls -lh "$tarball"

buildkite-agent artifact upload "$tarball"

exit "$test_status"
