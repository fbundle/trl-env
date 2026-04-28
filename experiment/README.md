# EXPERIMENT

## HOW TO SSH INTO A NODE RUNNING JOB

- submit job

```shell
echo "sleep 28800" | qsub -P $PBS_PROJECT -q normal -l select=1:ngpus=1 -l walltime=07:50:00 
```

- get hostname

```shell
qstat -ans <job_id> # or qstat -f <job_id>
```

- ssh

```shell
PBS_JOBID=<job_id> <hostname>
```