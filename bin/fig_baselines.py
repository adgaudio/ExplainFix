import pandas as pd
import glob
import re
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from subprocess import check_output
plt.rcParams.update({'font.size': 18,
                     'legend.title_fontsize': 16,
                     'legend.fontsize': 15,
                     'axes.labelsize': 16,
                     'lines.markersize': 18,
                     })



# chexpert baselines
fully_learned_baselines = {f'{mdl} {fromscratch}': check_output(
    f'cat results/C{{A,}}8-*{mdl.lower()}*{fromscratch.lower().replace("-","")}*unmodified_baseline*/eval_CheXpert_*d.csv|grep test_roc_auc_MEAN | cut -d, -f2', shell=True)
    for mdl in {'DenseNet', 'ResNet', 'EfficientNet'} for fromscratch in {'From-Scratch', 'Pre-Trained'}}
fully_learned_baselines = {
    mdl: np.array([float(x) for x in out.decode().strip().split('\n')])
    for mdl, out in fully_learned_baselines.items()}
fully_learned_baselines = pd.DataFrame(fully_learned_baselines)  # will fail if missing data
f = fully_learned_baselines
# chexpert baseline plot for appendix.  it's out of place here, but works.
fig, ax = plt.subplots(figsize=(12,10))
ax = sns.boxenplot(data=f, ax=ax, saturation=1, color='white', showfliers=False)
ax = sns.stripplot(data=f, ax=ax, jitter=False, size=15)
ax.hlines(.863, 0, 5, label="Existing Baseline", color='gray', linestyle='dashed')
ax.tick_params(axis='x', rotation=15)
ax.set_ylabel('Test Set Mean ROC AUC')
fig.savefig('chexpert_baselines.png', bbox_inches='tight')


# bbbc038v1 baselines
# todo: compare to GMM
# compare to deeplab

from os.path import join, basename
import glob
bbbc_data = {}
for dir in glob.glob('results/BB12*'):
#  for dir in glob.glob('bridges_results/BB12*'):
    print(dir)
    dfs = [pd.read_csv(fp).rename({'epoch': 'Epoch'}, axis=1).set_index('Epoch')
           for fp in glob.glob(join(dir, 'log/*perf.csv'))]
    # todo: 150
    dfs = [df['val_dice_'].rename("Test Set Dice")
     for df in dfs]
    # these models were evaluated on stage2 test set
    df = pd.concat(dfs)
    model = basename(dir).replace('BE12-unmodified_baseline-', '').replace('BB12-baseline-', '').replace('-BBBC038v1_stage2', '').title()
    bbbc_data[model] = df
df = pd.DataFrame(bbbc_data)
df = df[[df.columns[0], df.columns[2], df.columns[1], *df.columns[3:]]]
fig, axs = plt.subplots(1, 1, figsize=(8,8), squeeze=False)
axs = axs.ravel()
print(df.head())
sns.boxplot(data=df.query('Epoch > 150'), ax=axs[0], showfliers=False, saturation=1, color='white')
#  ax.hlines(.8534, 0, 300, label='GMM, manually tuned')
axs[0].tick_params(axis='x', rotation=5)
axs[0].set_ylabel('Test Set Dice')
#  axs[1].set_ylabel('Test Set Dice')
#  axs[1].set_ylim(.5, .9)
#  sns.lineplot(data=df, ax=axs[1])
#  fig.tight_layout()
fig.savefig('bbbc038v1_baselines.png', bbox_inches='tight')
plt.show()
