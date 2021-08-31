#!/usr/bin/env bash
# a script (run on the local computer) to reproduce our experiments.

set -e
set -u

cd "$(dirname "$(dirname "$(realpath "$0")")")"
# pwd

. ./bin/bash_lib.sh
# lockfile_ignore=true  # lockfile disabled due to pytorch autograd bug causes python 3.9 to crash on exit "terminate called without an active exception"
# lockfile_maxsuccesses=5
# lockfile_maxconcurrent=6
lockfile_maxfailures=3


V=12

# # Experiment 1:  100\% fixed filters
lockfile_maxsuccesses=6
lockfile_maxconcurrent=1
lockfile_maxfailures=7
(
# set -e
set -u

# NOTE: I think all of these models could be trained MUCH faster if the dataset
# was preprocessed before training rather than on-the-fly, but I didn't do that.

# baselines: compare unetd architecture to others
for x in `seq 1 3`; do
for mdl in unetD_small deeplabv3plus_mobilenet deeplabv3plus_resnet50 ; do
for mm in unmodified_baseline ; do
for dset in BBBC038v1_stage2 ; do
  experiment_id="BB$V-baseline-$mdl-$dset"
  echo run $experiment_id python bin/train.py --experiment-id $experiment_id --model "$mdl" --model-mode "$mm" --dset "$dset --do-this train --epochs 300"
done ; done; done
done

# main result, for 300 epochs.  unmodified baseline should eventually catch up.
for x in 1 2 3 4 5 6 ; do
  # for mm in "spatial_100%_SVDsteering_b7" "spatial_100%_ones" ; do # "spatial_100%_SVDsteeringC" "spatial_100%_SVDsteeringNC" "spatial_100%_DCT2steering" "spatial_100%_unchanged" "spatial_100%_psine" "spatial_100%_haar" "spatial_100%_DCT2" "unmodified_baseline" ; do  # "spatial_100%_kuniform" 
  for model in unetD_small ; do
  for mm in "unmodified_baseline" "spatial_100%_unchanged" "spatial_100%_ghaarA" "spatial_100%_ones" "spatial_100%_psine" "spatial_100%_DCT2" ; do
    run_id="BE${V}-$mm-$model"
    # BE11: in-distribution dataset
    # echo $run_id "python bin/train.py --experiment-id "$run_id" --model-mode "$mm" --dset-augmentations-train v2 --loss BCEWithLogitsLoss --model $model --epochs 300 --do-this train visualize --dset BBBC038v1 "
    # BE12: out-of-distribution dataset
    echo $run_id "python bin/train.py --experiment-id "$run_id" --model-mode "$mm" --dset-augmentations-train v2 --loss BCEWithLogitsLoss --model $model --epochs 300 --do-this train visualize --dset BBBC038v1_stage2 "
  done
  done
done

# main result, higher learning rate.
  for model in unetD_small ; do
  # for model in deeplabv3plus_resnet50 ; do
  for mm in "unmodified_baseline" "spatial_100%_unchanged" "spatial_100%_ghaarA" "spatial_100%_ones" "spatial_100%_psine" "spatial_100%_DCT2" ; do
    run_id="BR${V}-$mm-$model"
#     # BE11:
    # echo $run_id "python bin/train.py --experiment-id "$run_id" --model-mode "$mm" --dset-augmentations-train v2 --loss BCEWithLogitsLoss --model $model --epochs 300 --do-this train visualize --dset BBBC038v1 "
#     # BE12:
    echo run $run_id "python bin/train.py --experiment-id "$run_id" --model-mode "$mm" --dset-augmentations-train v2 --loss BCEWithLogitsLoss --model $model --epochs 300 --do-this train visualize --dset BBBC038v1_stage2 "
  done
  done

) # | run_gpus 1
