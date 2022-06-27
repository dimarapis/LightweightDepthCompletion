#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J auto_lambda
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request GB of system-memory
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u dimara@elektro.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

# Load the cuda module
module load numpy/1.21.1-python-3.8.11-openblas-0.3.17
module load cuda/10.2
alias python="python3"

python main.py -m "RGBD, untrained encoder, no custom init, train KITTI_600, test_KITTI_400"


## submit by using: bsub < job_run.sh
