#!/usr/bin/env bash
# a script (run on the local computer) to reproduce our experiments.

set -e
set -u
# set -o pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
# pwd

. ./bin/bash_lib.sh

# lockfile_ignore=true  # lockfile disabled due to pytorch autograd bug causes python 3.9 to crash on exit "terminate called without an active exception"
lockfile_maxsuccesses=1
lockfile_maxconcurrent=5
lockfile_maxfailures=1

V=8  # experiment version number


function quickrun() {
  local prefix="$1"
  local mdl="$2"
  local mm="$3"
  local dset="$4"
  local epochs="${5:-40}"
  local aug='none'
  local loss='chexpert_focal:1'
  local dothis='train eval visualize'
  # local dothis='eval --restore-checkpoint epoch_40.pth'
  local eval_dset='CheXpert_Small_L_valid'
  # local eval_dset='CheXpert_Small_D_valid'
  run_id="$prefix-$mdl-$mm-$dset-$loss"

  # tmp hack: only run jobs that didn't start
  # if [ 0 -ne "$(find ./results/$run_id/log -type f 2>/dev/null | wc -l)" ] ; then
    # return
  # fi
  # TMP hack: only run jobs lacking a checkpoint
  # if [ 0 -ne "$(find ./results/$run_id/checkpoints/epoch_40.pth -type f 2>/dev/null | wc -l)" ] ; then
    # return
  # fi
  echo "$run_id" python bin/train.py --experiment-id "$run_id" --model "$mdl" --model-mode "$mm" --loss "$loss" --dset "$dset" --dset-augmentations-train "$aug" --eval-dset "$eval_dset" --do-this $dothis --epochs $epochs
}

function quickshell() {
  local prefix="$1"
  local mdl="$2"
  local mm="$3"
  local dset="$4"
  local aug='none'
  local loss='chexpert_focal:1'
  local checkpoint='epoch_40.pth'
  local eval_dset='CheXpert_Small_L_valid'
  run_id="$prefix-$mdl-$mm-$dset-$loss"
  echo "--experiment-id "$run_id" --model "$mdl" --model-mode "$mm" --loss "$loss" --dset "$dset" --dset-augmentations-train "$aug" --do-this shell --restore-checkpoint "$checkpoint" "
  python bin/train.py --experiment-id "$run_id" --model "$mdl" --model-mode "$mm" --loss "$loss" --dset "$dset" --dset-augmentations-train "$aug" --eval-dset "$eval_dset" --do-this shell --restore-checkpoint "$checkpoint" 
}

# (

# get some baslines using classical wavelet methods.  They perform so poorly I didn't consider them in paper.
# quickrun "CB$V" haarlogistic:5 unmodified_baseline CheXpert_Small_L_15k_per_epoch 
# quickrun "CB$V" dualtreelogistic:5 unmodified_baseline CheXpert_Small_L_15k_per_epoch 
# quickrun "CB$V" scatteringlogistic:5 unmodified_baseline CheXpert_Small_L_15k_per_epoch 

# ) | run_gpus 3
(
set -e
set -u
set -o pipefail

# main ablative experiment: is performance affected by filter type or model architecture?
for x in `seq 1 $lockfile_maxsuccesses` ; do
for pt in "pretrained" "fromscratch" ; do
for mdl in "efficientnet-b0:5:$pt" "densenet121:5:$pt"  "resnet50:5:$pt"; do
for mm in "spatial_100%_GuidedSteer" "spatial_100%_DCT2" "unmodified_baseline" "spatial_100%_ghaarA" "spatial_100%_haar" "spatial_100%_psine" "spatial_100%_unchanged" "spatial_100%_SVDsteeringC" "spatial_100%_SVDsteeringNC" "spatial_100%_SVDsteeringNC_kde" ; do
# for mm in 'spatial_100%_GHaar2.ms' 'spatial_100%_GHaarR.ms' 'spatial_100%_PsineR.ms' 'spatial_100%_GHaar4.s1' 'spatial_100%_GHaar4.s2' ; do
for dset in CheXpert_Small_L_15k_per_epoch ; do
  quickrun C$V "$mdl" "$mm" "$dset"
  # run_id="debug-$mdl-$mm-$dset-$loss"
  # echo $run_id "python bin/train.py --epochs 1 --experiment-id "$run_id" --model "$mdl" --model-mode "$mm" --loss "$loss" --dset "CheXpert_Small-debug" --dset-augmentations-train "$aug" "
done ; done ; done ; done ; done
# ) | run_gpus 1

# ablative experiment: increasing model size
dset="CheXpert_Small_L_15k_per_epoch"
for mdl in "efficientnet-b5:5:fromscratch" "efficientnet-b0:5:fromscratch" "efficientnet-b2:5:fromscratch" "efficientnet-b3:5:fromscratch" "efficientnet-b1:5:fromscratch" "efficientnet-b4:5:fromscratch" ; do
for mm in "unmodified_baseline" "spatial_100%_unchanged" "spatial_100%_GuidedSteer" "spatial_100%_psine"; do
  quickrun "CM$V" "$mdl" "$mm" "$dset" 80
done ; done
# )  | run_gpus 1

# more baseline checkpoints for a zero weight experiment and for appendix
for iter in `seq 1 4` ; do
for pt in "pretrained" "fromscratch" ; do
for mdl in "efficientnet-b0:5:$pt" "densenet121:5:$pt"  "resnet50:5:$pt"; do
for mm in 'unmodified_baseline' ; do
for dset in CheXpert_Small_L_15k_per_epoch ; do
  quickrun CA$V-$iter "$mdl" "$mm" "$dset"
done ; done ; done ; done ; done

# ) | run_gpus 1

# Zero Weight Pruning models:
# some sample untrained but initialized imagenet models for zero weight pruning experiment
for mdl in efficientnet-b0; do # resnet50 densenet121 ; do
quickrun Z$V "$mdl:5:pretrained" "unmodified_baseline" CheXpert_Small_L_15k_per_epoch 0  --do-this save_checkpoint
done

) #| run_gpus 1
