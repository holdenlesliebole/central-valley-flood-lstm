#!/usr/bin/env bash
# Reconstruct the vendored upstream framework exactly as used in this project.
#
# Upstream: github.com/google-research/flood-forecasting (Apache-2.0),
# pinned to the commit this work was built against, plus one local patch
# (anonymous GCS reads for the public Caravan-MultiMet bucket) and the
# California basin lists / macOS conda env this project adds.
set -euo pipefail

cd "$(dirname "$0")/.."
PIN=ad122cb

git clone https://github.com/google-research/flood-forecasting.git \
    central_valley_floodforecasting
cd central_valley_floodforecasting
git checkout "$PIN"
git apply ../upstream/multimet-anon-gcs.patch
git apply ../upstream/tester-validation-zarr.patch
cp ../upstream/ca-basins-expanded.txt ../upstream/ca-sierra-basins.txt .
cp ../upstream/conda-macos.yml environments/

echo "Done. Next: build the env (see METHODS.md section 1) and 'pip install -e .'"
