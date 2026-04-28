#!/usr/bin/env bash

source .env
if [[ -v PBS_PROJECT ]]; then
    echo "sleep 86400" | qsub -P $PBS_PROJECT -q normal -l select=1:ngpus=1 -l walltime=23:50:00 
else
    echo "PBS_PROJECT not set"
fi