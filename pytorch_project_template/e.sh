#!/usr/bin/env bash
export PNAME="vanilla_pytorch_project_template"
export ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
echo "Welcome to $PNAME rooted at $ROOT"
echo "-"

# Activates conda environment
source activate ${PNAME}

# Configures paths. Adapt to your needs!
export PYTHONPATH=$ROOT:$PYTHONPATH
export DATA_DIR=$ROOT/data
export RESULTS_DIR=$ROOT/results

# Optional: enables plotting in iterm
# export MPLBACKEND="module://itermplot"

# Switches off importing out of environment packages
export PYTHONNOUSERSITE=1

if [ ! -d "${DATA_DIR}" ]; then
  echo "Creating ${DATA_DIR}"
  mkdir -p ${DATA_DIR}
fi

if [ ! -d "${RESULTS_DIR}" ]; then
  echo "Creating ${RESULTS_DIR}"
  mkdir -p ${RESULTS_DIR}
fi
