!#/usr/bin/env bash

# First we donwload the artifacts from the current PR build
buildkite-agent artifact download "*.txt" .

export MAIN_BUILD_NUM=45 # $(buildkite-agent meta-data get "latest-main-build-number")
export MAIN_BUILD_JOB=01966378-3b39-42b4-b65b-0af1a9b04892 # $(buildkite-agent meta-data get "latest-main-job-number")

echo $MAIN_BUILD_NUM
echo $MAIN_BUILD_JOB

# Then we fetch the artifacts from the main branch build
curl -s -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
"https://api.buildkite.com/v2/organizations/clima/pipelines/oceananigans-benchmarks-1/builds/${MAIN_BUILD_NUM}/jobs/${MAIN_BUILD_JOB}/artifacts" \
-o artifacts.json

echo "🔍 artifacts.json contents:"
cat artifacts.json

# Comparing the artifacts, creating the diff files and uploading them
for file in $(jq -r '.[] | select(.filename | endswith(".txt")) | .filename' artifacts.json); do
echo "Downloading ${file} from main..."
url=$(jq -r --arg filename "${file}" '.[] | select(.filename == ${filename}) | .download_url' artifacts.json)
curl -s -L "$url" -o "baseline_${file}"

echo "Comparing $file with baseline..."
if [ -f "${file}" ]; then
    diff -u "baseline_$file" "${file}" > "diff_${file}" || true
else
    echo "${file} not found in PR build. Skipping diff." > "diff_${file}"
fi
done

echo "Uploading diffs..."
buildkite-agent artifact upload "diff_*.txt"

echo "Annotating summary..."
for diff in diff_*.txt; do
    buildkite-agent annotate --style info --context "${diff}" < "${diff}"
done