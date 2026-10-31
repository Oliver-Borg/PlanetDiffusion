#!/bin/bash
# Creates the conda environment in ./.conda-env, installing Miniforge locally first if conda is not available.
set -e
cd "$(dirname "$(readlink -f "$0")")"

ENV_DIR="$(pwd)/.conda-env"
MINIFORGE_DIR="$(pwd)/.miniforge"

if command -v conda &> /dev/null; then
  CONDA="conda"
elif [ -x "$MINIFORGE_DIR/bin/conda" ]; then
  CONDA="$MINIFORGE_DIR/bin/conda"
else
  case "$(uname)-$(uname -m)" in
    Linux-x86_64)  INSTALLER="Miniforge3-Linux-x86_64.sh" ;;
    Linux-aarch64) INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
    *) echo "Unsupported platform $(uname)-$(uname -m). Install conda manually and rerun."; exit 1 ;;
  esac
  echo "conda not found, installing Miniforge to $MINIFORGE_DIR"
  curl -L -o "/tmp/$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER"
  bash "/tmp/$INSTALLER" -b -p "$MINIFORGE_DIR"
  rm "/tmp/$INSTALLER"
  CONDA="$MINIFORGE_DIR/bin/conda"
fi

if [ -d "$ENV_DIR" ]; then
  "$CONDA" env update -y -p "$ENV_DIR" -f environment.yml --prune
else
  "$CONDA" env create -y -p "$ENV_DIR" -f environment.yml
fi

echo "Setup complete. Run the interface with ./run.sh, or activate the environment with:"
echo "    $CONDA activate $ENV_DIR"
