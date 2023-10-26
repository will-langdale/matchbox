#! /bin/bash
# Purpose: A shell script to run the initial setup of the project in a single command
# Usage: . setup.sh

eval "$(conda shell.bash hook)"
# Deactivating the environment allows "make environment" to delete it
conda deactivate
make environment
conda activate company_matching
make requirements