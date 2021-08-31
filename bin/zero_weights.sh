# # Zero Weights, Experiment 1
# # to evaluate performance loss as spatial filters are progressively removed from (replace with zeroes) a trained model
echo python bin/zero_weights.py --e1 'baseline'

# # Zero Weights, Experiment 2 models:
# # --> experiments showing spatial filters are...
for prefix in 1_ 2_ 3_ 4_ 5_ 6_ ; do
  cat <<EOF
  # unnecessary for inference (these models are fine-tuned from epoch 41 after removing spatial filter weights.)
  python bin/zero_weights.py --e2 'baseline.*effic.*pret' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'baseline.*dense.*pret' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'baseline.*resn.*pret' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'Unchanged.*effic.*pret' --e2-pct 85  --savefile-prefix $prefix  # was 90 but rerunning with 85
  python bin/zero_weights.py --e2 'Unchanged.*dense.*pret' --e2-pct 95  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'Unchanged.*resn.*pret' --e2-pct 99  --savefile-prefix $prefix

  # unnecessary for training and inference (these models are trained from epoch 1.  Random Fixed on from scratch is equivalent to Unchanged on fromscratch model )
  python bin/zero_weights.py --e2 'baseline.*effic.*fromsc' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'baseline.*dense.*fromsc' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'baseline.*resn.*fromsc' --e2-pct 0  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'Random Fixed.*effic.*fromsc' --e2-pct 80  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'Random Fixed.*dense.*fromsc' --e2-pct 85  --savefile-prefix $prefix
  python bin/zero_weights.py --e2 'Random Fixed.*resn.*fromsc' --e2-pct 90  --savefile-prefix $prefix
EOF
done  #  | parallel -j 6


# Experiment 3: Zero Weights + Pruning Experiments.
# # --> Train pruned models to show they have same predictive perf.
# --> note: ./bin/chexpert_experiments.sh in section "Zero Weight Pruning models" initializes the needed models and training configurations.
# --> pruned resnet
# resnet
for x in `seq 1 5` ; do 
  cat <<EOF
python bin/zero_weights.py --e2 'Pruned Baseline.*resnet.*pret' --e2-pct 99.9 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 '(ImageNet).*resnet.*pret'      --e2-pct 99.9 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 'Pruned Baseline.*resnet.*pret' --e2-pct 99 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 '(ImageNet).*resnet.*pret'      --e2-pct 99 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 'Pruned Baseline.*resnet.*pret' --e2-pct 90 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 '(ImageNet).*resnet.*pret'      --e2-pct 90 --e2-prune --savefile-prefix "${x}_"
# densenet
python bin/zero_weights.py --e2 '(ImageNet).*densenet.*pret'      --e2-pct 90 --e2-prune --savefile-prefix "${x}_"
python bin/zero_weights.py --e2 'Pruned Baseline.*densenet.*pret' --e2-pct 90 --e2-prune --savefile-prefix "${x}_"
# note: efficientnet is not possible to prune right now because of the implementation using parameterized functions ( T.conv2d ) directly rather than modules (T.nn.Conv2d).  I didn't figure out how to support this setting.
EOF
done

# Experiment 4: Timing experiments for fixed and fixed+pruned models
# pruned, fixed
cat <<EOF
python bin/zero_weights.py --e2 '(ImageNet).*resnet.*pret'               --e2-pct 99.9 99 90 --e2-prune --savefile-prefix "timing" --e2-epochs 20
python bin/zero_weights.py --e2 '(ImageNet).*densenet.*pret'              --e2-pct 80 90      --e2-prune --savefile-prefix "timing" --e2-epochs 20
# not pruned, learned
python bin/zero_weights.py --e2 'baseline.*(resnet|densenet).*pret'      --e2-pct 0                     --savefile-prefix "timing" --e2-epochs 20
# not pruned, fixed
python bin/zero_weights.py --e2 'ImageNet.*(resnet|densenet).*pret'      --e2-pct 0                     --savefile-prefix "timing" --e2-epochs 20
# efficientnet
python bin/zero_weights.py --e2 '(ImageNet|baseline).*(efficientnet).*pret'      --e2-pct 0                     --savefile-prefix "timing" --e2-epochs 20
# pruned, learned.  maybe do these?  it's a bit redundant
python bin/zero_weights.py --e2 '(Pruned Baseline).*resnet.*pret'               --e2-pct 99.9 99 90 --e2-prune --savefile-prefix "timing" --e2-epochs 20
python bin/zero_weights.py --e2 '(Pruned Baseline).*densenet.*pret'              --e2-pct 80 90      --e2-prune --savefile-prefix "timing" --e2-epochs 20
EOF
