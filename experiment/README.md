# EXPERIMENT

## HOW TO SSH INTO A NODE RUNNING JOB

- submit job

```shell
echo "sleep 86400" | qsub -P $PBS_PROJECT -q normal -l select=1:ngpus=1 -l walltime=23:50:00 
```

- get hostname

```shell
qstat -f <job_id> | grep exec_host
```

- ssh

```shell
PBS_JOBID=<job_id> <hostname>
```
