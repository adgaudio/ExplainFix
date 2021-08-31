# Navigate to root of repository
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# Obtain Datasets:
mkdir ./data
# download and extract datasets to make these directories:
function check_exists(){
  test -e "$1" || echo "Missing Dataset:  $1"
}
check_exists ./data/CheXpert-v1.0-small/
#      # $ ls ./data/CheXpert-v1.0-small
#      # train  train.csv  valid  valid.csv
check_exists ./data/BBBC038v1_microscopy/
#       # ls ./data/BBBC038v1_microscopy
#       # kaggle_data_science_bowl_2018  stage1_test.zip              stage1_train.zip
#       # stage1_solution.csv            stage1_train                 stage2_solution_final.csv
#       # stage1_solution.csv.zip        stage1_train_labels.csv      stage2_test_final
#       # stage1_test                    stage1_train_labels.csv.zip  stage2_test_final.zip


# Train Models:

# For GuidedSteer:  save the probability distributions to disk
python bin/svd_steering.py

# this spits out a huge list of 300+ commands to run in command-line.  Some commands train multiple models. Each command requires one gpu.
# --> results (logs, generated data, checkpoints) end up organized in ./results.
. ./bin/bash_lib.sh
(
./bin/bbbc038v1_experiments.sh
./bin/chexpert_experiments.sh
./bin/zero_weights.sh
) # | run_gpus 1


# plotting (this should cover all figures in the paper.)
python bin/fig_baselines.py
python bin/fig_dct_basis.py
python bin/fig_ghaar_construction.py
python bin/fig_heatmap_C8_experiments.py
python bin/fig_table_num_params_bbbc_baselines.sh
python bin/zero_weights_plots --exp1 
python bin/zero_weights_plots --exp2
python bin/zero_weights_plots --exp3
python bin/zero_weights_plots --exp4
python bin/zero_weights_plots --exp5
# --> NOTE about replicating the BBBC experiment plots: In the
# bbbc038v1_experiments.sh, I used earlier "versions", V=10
# and V=11, for appendix figures (BBBC learning rate and the
# in-distribution test).  To recreate the figures, you'd need make V=10 or V=11
# and then tweak some hardcoded commented-out settings in the plotting file:  fig_bbbc038v1_plots.py
# Those two appendix figures would therefore probably be challenging to re-create.
simplepytorch_plot BE12 --mode 3 --no-plot -c none < ./bin/fig_bbbc038v1_plots.py
simplepytorch_plot BE11 --mode 3 --no-plot -c none < ./bin/fig_bbbc038v1_plots.py
