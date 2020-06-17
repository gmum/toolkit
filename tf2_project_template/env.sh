export ENVCALLED=1
export PNAME=tf2_project_template

export ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
echo "Welcome to $PNAME rooted at $ROOT"
echo "-"

# Activates conda environment
source activate $PNAME

# Configures paths. Adapt to your needs!
export PYTHONPATH=$ROOT:$PYTHONPATH
export DATA_DIR=$ROOT/data
export RESULTS_DIR=$ROOT/results

# Optional: enables plotting in iterm
export MPLBACKEND="module://itermplot"

# Optional: enables copying figures to Dropbox (useful for managing figures for paper)
export DROPBOXTOKEN=FILLME

# Optional: enables integration with Neptune
export NEPTUNE_TOKEN=FILLME
export NEPTUNE_PROJECT=tf2projecttemplate
export NEPTUNE_USER=FILLME
# export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE=1 # Uncomment if you have issues with SSL

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