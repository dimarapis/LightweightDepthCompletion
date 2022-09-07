#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J auto_lambda
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request GB of system-memory
#BSUB -R "rusage[mem=16GB]"
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

#python main.py --network-model SparseGuidedDepth -m "BATCHOVERFIT - NN - RGB input - encoderpretrained" --wandblogger WANDBLOGGER --wandbrunname "BATCHOVERFIT-NN-RGB-untrained" 
#python main.py --network-model SparseAndRGBGuidedDepth -m "visualize kitti GT" --wandblogger WANDBLOGGER --wandbrunname "visualizekittiGT"
#python main.py --network-model SparseAndRGBGuidedDepth -m "NN 500/100 dataset - RGBD input - encoderpretrained" --wandblogger WANDBLOGGER --wandbrunname "NN-RGBD-encoderpretrained-2"
#python main.py --network-model SparseGuidedDepth -m "NN 500/100 dataset - D input - encoderpretrained" --wandblogger WANDBLOGGER --wandbrunname "NN-D-encoder-pretrained-7"
#python main.py --network-model AuxSparseGuidedDepth -m "DepthCorrectionRefinement" --pretrained PRETRAINED --wandblogger WANDBLOGGER --wandbrunname DepthCorrectionRefineme
#python main.py --network-model AuxSparseGuidedDepth -m DepthCorrectionRefinement --pretrained PRETRAINED --wandblogger WANDBLOGGER --wandbrunname DepthCorrectionRefinementlrscheduler 

#python main.py --network-model AuxSparseGuidedDepth -m "Testing consistency with simple model" --pretrained PRETRAINED --wandblogger WANDBLOGGER --wandbrunname Test_2 
#python main.py --network-model AuxSparseGuidedDepth -m "Testing consistency with simple model" --pretrained PRETRAINED --wandblogger WANDBLOGGER --wandbrunname deletetest
## submit by using: bsub < job_run.sh

#python main.py --network-model AuxSparseGuidedDepth -m "Testing PENETC2 efficacy on KITTI" --wandblogger WANDBLOGGER --wandbrunname "KITTI_dilatedCSPNRefinement" --pretrained PRETRAINED --dataset kitti
#python main.py --network-model AuxSparseGuidedDepth -m "basemodel trained on kitti, finetuned on NN - initiallr 1-e5" --wandblogger WANDBLOGGER --wandbrunname NN_basemodel_finetune_lr1e-5 --pretrained PRETRAINED --dataset nn
#python main_ref.py --network-model AuxSparseGuidedDepth -m "basemodel finetuned alone, training only refinement lr 1-e6" --wandblogger WANDBLOGGER --wandbrunname kitti_encoder_finetuned_training_ref --pretrained PRETRAINED --dataset kitti


#python main_singlemodel.py --dataset nyuv2 --network GuideDepth --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_50k_GuideDepth --batch-size 8 --message "Training GuideDepth with 50k to see how close it goes to original"
#python main_singlemodel.py --dataset nyuv2 --network DecnetModule --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_2000_Decnet500 --batch-size 8 --message "Training DecnetModule with 2k minibatch and 500 sparse"
#python main_singlemodel.py --dataset nyuv2 --network DecnetModule --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_2000_DecnetFULLSPARSE --batch-size 4 --message "Training DecnetModule with 2k minibatch and full sparse"
#python main_singlemodel.py --dataset nyuv2 --network DecnetModule --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_50k_Decnet500 --batch-size 8 --message "Training DecnetModule with 50k nyuv2 and 500 sparse"
#python main_singlemodel.py --dataset nyuv2 --network DecnetModule --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_2k_Decnet500 --batch-size 8 --message "Training DecnetModule with all 2k  and 500 sparse"

#python main_singlemodel.py --dataset nyuv2 --network GuideDepth --learning-rate 0.0001 --wandblogger TRUE --wandbrunname NYU_50k_GuideDepthAugmentL1 --batch-size 8 --message "MaskedL1error" --augment TRUE --pretrained TRUE
python main_singlemodel.py --dataset nyuv2 --networkmodel s2d --learning-rate 0.00001 --wandblogger TRUE --wandbrunname NYU_50k_s2d_sparsities_finetunelr05 --batch-size 16 --message "50k_s2d_sparsities_finetunewith lr 0.00001" --augment TRUE --sparsities TRUE --pretrained TRUE