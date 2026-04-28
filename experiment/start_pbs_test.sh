#!/usr/bin/env bash

source .env
if [[ -v PBS_PROJECT ]]; then
    qsub -I -P $PBS_PROJECT -q normal -l select=1:ngpus=1 -l walltime=07:50:00
else
    echo "PBS_PROJECT not set"
fi