"""
Heatmap plot for presentation, summarizing C8 performance
"""
import pandas as pd
import glob
import re
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
#  from scipy.stats import ttest_rel as wilcoxon
import seaborn as sns
from matplotlib.colors import CenteredNorm

plt.rcParams.update({'font.size': 18})


def parse(id_):
    dct = re.match(r'.*?C.?8-(?P<model>(efficientnet-b\d|densenet\d+|resnet\d+)):(5|14):(?P<pretrained>(pretrained|fromscratch))-(?P<method>.*?)-.*', id_).groupdict()
    return (dct['model'].replace('densenet', 'DenseNet').replace('resnet', 'ResNet').replace('efficientnet', 'EfficientNet'),
            dct['pretrained'].replace('pretrained', 'PreTrained').replace('fromscratch', 'FromScratch'), dct['method'])


def get_df(*glob_exprs):
    fps = list(sorted({x for glob_expr in glob_exprs for x in glob.glob(glob_expr)}))
    ignore_these = {'kuniform',  # same as "unchanged" for fromscratch models, and need to make space for DCT2 filters.
                    'GHaar4.', 'GHaar2.', 'GHaarR.', 'PsineR.',  # a side experiment with partial results
                    'ghaar4', 'ghaar-', 'ghaarN', '_haar-',
                    'GuidedSteerC', 'SVDsteeringC', 'SVDsteeringNC', 'SVDsteeringNC_kde',
                    'DCT2steering'  # method works, but violates diagnoalization assumption and otherwise nearly same as svdsteering
                    }
    fps = [fp for fp in fps if not any(x in fp for x in ignore_these)]
    print(fps)
    dct = {fp: pd.read_csv(fp, header=None, index_col=0, names=['metric', 'value'])['value'] for fp in fps}
    dct = {parse(fp): v
           for fp, v in dct.items()}
    cols = ['test_roc_auc_Atelectasis',
            'test_roc_auc_Cardiomegaly',
            'test_roc_auc_Consolidation',
            'test_roc_auc_Edema',
            'test_roc_auc_Pleural Effusion',
            'test_roc_auc_MEAN',
            ]
    df = pd.concat(dct).unstack('metric')[cols].astype('float')
    df['test_roc_auc_MEDIAN'] = df[cols[:-1]].median(1)
    #  df['test_roc_auc_MEAN'] = df[cols[:-1]].mean(1)   # sanity checked
    df.index = df.index.set_levels(
        [x[0].upper() + x[1:] for x in df.index.levels[2]\
         .str.replace('spatial_100%_', '')\
         .str.replace('unmodified_baseline', 'Learned Baseline')\
         .str.replace('ghaarA', 'GHaar')
         ], level=2)
    return df


def plot_heatmap(df, ax):
    g = sns.heatmap(
        df['test_roc_auc_MEAN'].unstack(level=-1),
        cmap='YlGn', annot=True, fmt='0.03g', ax=ax, cbar=False,
        norm=CenteredNorm(.866),
    )
    # tODO: fix coloring by row.
    g.set_yticklabels(g.get_yticklabels(), rotation=25)
    g.set_xticklabels(g.get_xticklabels(), rotation=15)
    g.set_ylabel('')
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=.9)


def plot_bar(df, ax):
    z = df['test_roc_auc_MEAN'].unstack(level=-1)
    z.index = [x[0][-2:] for x in z.index]
    assert z.columns[-1] == 'Learned Baseline'
    z.iloc[:,:-1].plot.line(ax=ax)
    z.iloc[:,:-1].plot.line(ax=ax, style='o', legend=False, color=['steelblue', 'darkorange', 'green'])
    z[['Learned Baseline']].plot.line(ax=ax, color='black', linewidth=3, alpha=.8)
    z['Learned Baseline'].plot.line(ax=ax, color='black', style='o', legend=False, alpha=.8)
    ax.set_xlabel('EfficientNet Model Size')
    ax.set_ylabel('Test Set Mean AUC ROC')
    plt.show(block=False)


#  import numpy as np
#  def plot_heatmap(df, ax):
    #  # adapted from https://stackoverflow.com/questions/60325792/seaborn-heatmap-color-by-row
    #  _df = df
    #  df = df['test_roc_auc_MEAN'].unstack(level=-1)
    #  N_cols = len(df.columns)
    #  for rowidx, row in enumerate(df.values):
        #  #  ax.imshow(np.vstack([row, row]), aspect='auto', extent=[-0.5,N_cols-0.5,rowidx,rowidx+1], cmap='RdYlGn', norm=CenteredNorm(row[-1]))
        #  new = np.zeros_like(df.values)
        #  new[rowidx] = df.values[rowidx]
        #  ax.imshow(new, cmap='RdYlGn', norm=CenteredNorm(df.values[rowidx][-1]))
        #  for colidx,val in enumerate(row):
            #  if np.isnan(val):
                #  continue
            #  ax.annotate(f'{val:.03g}', xy=(colidx,rowidx+0.5), ha='center', va='center', color='black')
    #  import IPython ; IPython.embed() ; import sys ; sys.exit()
    #  import sys ; sys.exit()

def significance_tests(df, thresh=.0001, fname_out=None):
    df = df.copy()
    # combine model architecture and pretraining/fromscratch distinction
    df.index = [df.index.map('{0[0]}-{0[1]}'.format), df.index.get_level_values(2)]
    # delete undesirable column for hypothesis tests
    del df['test_roc_auc_MEAN']
    del df['test_roc_auc_MEDIAN']

    fixedfilter_method = [
        x for x in df.index.levels[1] if x != 'Learned Baseline']

    # Compute paired significance test comparing each method to baseline,
    # compute one test for each model, sampling across the predictive tasks.
    tbl1 = []
    tbl1header = ['Model', '100% Fixed Spatial Filter Method', 'Significant Difference? (2-sided p-value)', 'Underperforming (1-sided p-value)', 'Outperforming (1-sided p-value)']
    for ffmethod in fixedfilter_method:
        #  print('\n', ffmethod, '\n', '\n')
        for mdlarch in df.index.levels[0]:
                x = df.loc[mdlarch].stack()
                baseline = x.loc['Learned Baseline']  # comparing across tasks
                # for exp in fixedfilter_method:
                try:
                    pval_diff_than_baseline = wilcoxon(
                        x.loc[ffmethod].values, baseline.values).pvalue
                    pval_worse_than_baseline = wilcoxon(
                        x.loc[ffmethod].values, baseline.values, alternative="less").pvalue
                    pval_better_than_baseline = wilcoxon(
                        x.loc[ffmethod].values, baseline.values, alternative="greater").pvalue
                except:
                    # hack
                    print('SKIP', ffmethod, mdlarch, 'significance_tests(...)')
                    pval_diff_than_baseline, pval_worse_than_baseline, pval_better_than_baseline = 0,0,0
                tbl1.append(
                    (mdlarch, ffmethod,
                     (f'{"YES" if pval_diff_than_baseline < thresh else "NO"}'
                      f' ({pval_diff_than_baseline:.05g})'),
                     (f'{"YES" if pval_worse_than_baseline < thresh else "NO"}'
                      f' ({pval_worse_than_baseline:.05g})'),
                     (f'{"YES" if pval_better_than_baseline < thresh else "NO"}'
                      f' ({pval_better_than_baseline:.05g})'),
                     ))
                #  print(
                    #  mdlarch, ffmethod, '\n',
                    #  wilcoxon(x.loc[ffmethod].values, baseline.values).pvalue,
                    #  wilcoxon(x.loc[ffmethod].values, baseline.values).pvalue,
                    #  'greater', wilcoxon(x.loc[ffmethod].values, baseline.values, alternative='greater').pvalue<thresh,
                    #  'less', wilcoxon(x.loc[ffmethod].values, baseline.values, alternative='less').pvalue<thresh,
                #  )
    del ffmethod, mdlarch, baseline, x, pval_diff_than_baseline, pval_worse_than_baseline
    tbl1 = pd.DataFrame(tbl1, columns=tbl1header)
    print(tbl1.to_string())

    # significance, one test for each task, sampling across the 6 models.
    tbl3 = []
    tbl3header = [
        'Task', '100% Fixed Spatial Filter Method',
        'Significant Difference? (2-sided p-value)',
        'Underperforming (1-sided p-value)',
        'Outperforming (1-sided p-value)']
    for ffmethod in fixedfilter_method:
        dftasks = df.stack().unstack(level=[0])
        for task in dftasks.index.levels[1]:
            baseline = dftasks.loc['Learned Baseline', task]  # comparing across models
            pval_diff_than_baseline = wilcoxon(dftasks.loc[ffmethod,task].values, baseline.values).pvalue
            pval_worse_than_baseline = wilcoxon(dftasks.loc[ffmethod,task].values, baseline.values, alternative="less").pvalue
            pval_better_than_baseline = wilcoxon(dftasks.loc[ffmethod,task].values, baseline.values, alternative="greater").pvalue
            tbl3.append(
                (task, ffmethod,
                 f'{"YES" if pval_diff_than_baseline < thresh else "NO"} ({pval_diff_than_baseline:.05g})',
                 f'{"YES" if pval_worse_than_baseline < thresh else "NO"} ({pval_worse_than_baseline:.05g})',
                 f'{"YES" if pval_better_than_baseline < thresh else "NO"} ({pval_better_than_baseline:.05g})',
                 ))
    tbl3 = pd.DataFrame(tbl3, columns=tbl3header)
    print(tbl3.sort_values(['Task', '100% Fixed Spatial Filter Method']).to_string())
    del ffmethod, dftasks, task, baseline, pval_diff_than_baseline, pval_worse_than_baseline

    # Compute paired significance test comparing each method to baseline,
    # considering as samples the models and tasks.
    tbl2 = []
    tbl2header = [
        'Method',
        f'Equal? (p)',
        #  f'Underperforming models? (pval<{thresh})',
        #  f'Underperforming tasks? (pval<{thresh})']
    ]
    baseline = df.xs('Learned Baseline', level=1).values.ravel()
    num_underperforming_models = tbl1\
        .groupby('100% Fixed Spatial Filter Method')\
        ['Underperforming (1-sided p-value)']\
        .apply(lambda x: f'{x.str.startswith("YES").sum()} / {x.shape[0]}')
    num_underperforming_tasks = tbl3\
        .groupby('100% Fixed Spatial Filter Method')\
        ['Underperforming (1-sided p-value)']\
        .apply(lambda x: f'{x.str.startswith("YES").sum()} / {x.shape[0]}')
    for exp in fixedfilter_method:
        try:
         pval_worse_than_baseline = wilcoxon(df.xs(exp, level=1).values.ravel(), baseline, alternative="less").pvalue
        except:
            # hack
            print('=====WARNING============')
            print('skip', exp, 'significance_test(...) output tbl2')
            print('-END-WARNING------------')
            continue
        tbl2.append((
            exp,
            f'{"NO" if pval_worse_than_baseline < thresh else "YES"} ({pval_worse_than_baseline:.02g})',
            #  num_underperforming_models.loc[exp],
            #  num_underperforming_tasks.loc[exp],
        ))
    tbl2 = pd.DataFrame(tbl2, columns=tbl2header)
    print(tbl2.to_string())

    latex = tbl2.to_latex(index=False)\
            .replace('YES', '\cellcolor{green!25}YES')\
            .replace('NO', '\cellcolor{red!25}NO')
    #  latex = r'\usepackage{colortbl}\n' + latex
    globals().update(dict(tbl1=tbl1, tbl3=tbl3))
    if fname_out:
        with open(fname_out, 'w') as fout:
            fout.write(latex)
        return tbl2



COLS =  ['DCT2', 'GHaar', 'Psine', 'Unchanged', 'GuidedSteer', 'Learned Baseline']
# ablative experiment: varying number of principle components in GuidedSteer CG8
fig, ax = plt.subplots(1,1,figsize=(14,12))
df = get_df('results/CG8-*/eval*d.csv', 'results/C8-den*unmodif*/eval*d.csv')
tbl = significance_tests(df, fname_out='./chexpert_table_significance_tests_CG8.tex')
plot_heatmap(df, ax=ax)
ax.set_title('Ablative Experiment: Removing Principle Components from Steered Initialization.')
fig.savefig('./chexpert_heatmap_CG8.png', bbox_inches='tight')

# main ablative experiment: varying fixed filters and model architecture
fig, ax = plt.subplots(1,1,figsize=(12,5))
df = get_df('results/C8-*/eval*d.csv')
df = df.reindex(COLS, axis=0, level=2)
tbl = significance_tests(df, fname_out='./chexpert_table_significance_tests_C8.tex')
plot_heatmap(df, ax=ax)
ax.set_title(
    #  'Fixed Filters Have Roughly Equal Predictive Performance to Baseline\n'
    'Test Set Mean AUC ROC, CheXpert U-ignore baseline')
fig.savefig('./chexpert_heatmap_C8.png', bbox_inches='tight')

#  # --> extended epochs models
fig, ax = plt.subplots(1,1,figsize=(14,12))
df = get_df('results/CE8-*/eval*d.csv')
df = df.reindex(COLS, axis=0, level=2)
tbl = significance_tests(df, fname_out='./chexpert_table_significance_tests_CE8.tex')
plot_heatmap(df, ax=ax)
ax.set_title('Ablative Experiment: Varying Model Architecture and Fixed Filters.\nMean AUC ROC across 6 leaderboard tasks on CheXpert Test Set')
fig.savefig('./chexpert_heatmap_CE8.png', bbox_inches='tight')

# ablative experiment: more output classes
#  fig, ax = plt.subplots(1,1,figsize=(14,12))
#  df = get_df('results/CT8-*/eval*d.csv')
#  df = df.reindex(COLS, axis=0, level=2)
#  plot_heatmap(df, ax=ax)
#  fig.axes[0].set_title('Ablative Experiment with More Output Classes.\nMean AUC ROC across 14 leaderboard tasks on CheXpert Test Set')
#  fig.savefig('./chexpert_heatmap_CT8.png', bbox_inches='tight')

#  ablative experiment: varying model size
df = get_df('results/CM8-*fromsc*/eval*d.csv')
df = df.reindex(COLS, axis=0, level=2)
tbl = significance_tests(df, fname_out='./chexpert_table_significance_tests_CM8.tex')
plot_heatmap(df, ax=ax)
fig.savefig('./chexpert_heatmap_CM8.png', bbox_inches='tight')
fig, ax = plt.subplots(1,1,figsize=(14,12))
fig.axes[0].set_title('Varying Model Size.\nMean AUC ROC across 6 leaderboard tasks on CheXpert Test Set')
#  ax.set_title("Robustness to Overfitting as Model Capacity Increases.")
fig, ax = plt.subplots(1,1,figsize=(10,3))
plot_bar(df, ax=ax)
ax.set_ylim(0,1)
fig.savefig('./chexpert_CM8_robust_overfitting.png', bbox_inches='tight')
#  ax.set_ylim(.75)

# ablative experiment: changing dataset size
#  fig, ax = plt.subplots(1,1,figsize=(14,12))
#  df = get_df('results/CD8-*/eval*d.csv')
#  df = df.reindex(COLS, axis=0, level=2)
#  plot_heatmap(df, ax=ax)
#  fig.axes[0].set_title('Ablative Experiment: Varying Training Dataset Size.\nMean AUC ROC across 6 leaderboard tasks on CheXpert Test Set')
#  fig.savefig('./chexpert_heatmap_CD8.png', bbox_inches='tight')

# ablative experiment: is there an optimal percentage of fixed filters?
#  fig, ax = plt.subplots(1,1,figsize=(8,8))
#  df = get_df('results/CP8-*/eval*d.csv')
#  df = df.reindex(COLS, axis=0, level=2)
#  plot_heatmap(df, ax=ax)
#  fig.axes[0].set_title('Varying Percentage Fixed Filters.\nMean AUC ROC across 6 leaderboard tasks on CheXpert Test Set')
#  fig.savefig('./chexpert_heatmap_CP8.png', bbox_inches='tight')


"""
Why we use 1-sided p-values:
    1-sided hypothesis tests have more power than 2-sided tests.
    More power to distinguish significance is better because the test has better
    ability to reject the hypothesis that the given fixed filter method gives the
    same performance as the baseline.
Why we use wilcoxon vs ttest
    Assume non-gaussian
    It is more likely to say a fixed method underperforms baseline, though both tests give nearly identical table (we verified empirically and 

idea:
    5 learning curve plot for the prediction tasks,
    once for each architecture --> 15 plots.
latex: YES: red, NO: green
"""
