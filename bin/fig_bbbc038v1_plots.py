# simplepytorch_plot BE12 --mode 3 --no-plot -c none < ./bin/fig_bbbc038v1_plots.py
# or
# %run -i bin/quick_plots_for_final_pres.py

import seaborn as sns
from matplotlib import rcParams
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, ks_2samp
rcParams.update({
    'figure.titlesize': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
})


df = cdfs.copy()
# tweak display 
_tmp = df.index.names
df.reset_index(inplace=True)
#  try:
#  tmp = df.run_id.str.extractall('(?P<tag>BE?10)-(?P<method>.*)').reset_index('match', drop=True)
    #  assert tmp.isna().values.sum() == 0
    #  assert tmp.size
    #  df = df.join(tmp.reset_index('match', drop=True))
#  except:
tmp = df.run_id.str.extractall('(?P<tag>B[ER]1[12])-(?P<method>.*?)-(?P<model>.*)').reset_index('match', drop=True)
df = df.join(tmp)
df['method'] = df['method']\
        .str.replace('spatial_100%_', '')\
        .str.replace('unmodified_baseline', 'Learned Baseline')\
        .str.replace('ghaarA', 'GHaar')\
        .apply(lambda rc: rc[:1].upper() + rc[1:])
df['run_id'] = df['method']# + '-' + df['model']
df.set_index(_tmp, inplace=True)
df['Dice'] = df['val_dice_']
del df['val_dice_']
fixed_filter_methods = ['Ones', 'DCT2', 'Unchanged', 'GHaar', 'Psine']
#  fixed_filter_methods = [x for x in df.index.levels[0] if x not in {'Learned Baseline', 'SVDsteering'}]


# Table: Winning ticket initializations
import numpy as np
import scipy.stats as st
def mean_ci(x):
    x = x['Dice'].values
    return pd.Series([np.mean(x), np.mean(x) - st.t.interval(alpha=.99, df=len(x)-1, loc=np.mean(x), scale=st.sem(x))[1]], index=['Test Set Dice', '99% CI'])
ep = 150
z = df.query(f'epoch >= {ep}')[['Dice']].groupby(['run_id', 'filename']).apply(mean_ci)
# table with error bars of the best run.
z_tbl = z.loc[z.groupby('run_id').apply(lambda x: x['Test Set Dice'].idxmax())]
# or plot with error bars
z = df.query(f'epoch >= {ep}').reset_index('epoch').loc[z_tbl.index].rename({'Dice': 'Test Set Dice'}, axis=1).reset_index()
order = ['Ones', 'DCT2', 'Psine', 'Unchanged', 'GHaar', 'Learned Baseline']
ax = sns.catplot(
    data=z.reset_index(), x='run_id', y='Test Set Dice', hue='run_id', ci=95,
    join=False, kind='point', order=order, height=4, aspect=3.0,
    palette=sns.color_palette(['gray', 'green', 'red',  'gray', 'green', 'green']),).axes[0,0]
ylim = ax.get_ylim()
ax.hlines(z_tbl.xs('Learned Baseline').values[0], *ax.get_xlim(), linestyle='dashed', color='gray', linewidth=1)
ax.set_ylim(ylim)
ax.set_xlabel('Method')
#  ax.legend(['Non-steered Fixed Filter Initialization', 'Steered Fixed Filter Initialization', 'Learned Baseline'], labelcolor=['gray', 'green', 'red'])
h,l = ax.get_legend_handles_labels()
ax.legend(h[:3], ['Non-Steered Fixed Initializations', 'Steered Fixed Initializations', 'Fully Learned Baseline'])
for i in range(len(order)):
    val = z_tbl.xs(order[i])['Test Set Dice']
    ax.text(i+.1, val.values[0], '%0.4g' % val, fontsize=20)
ax.figure.tight_layout()
#  ax.figure.savefig('e1_BE10-winning_ticket.png', bbox_inches='tight')
#  ax.figure.savefig('e1_BE11-winning_ticket.png', bbox_inches='tight')
#  ax.figure.savefig('e1_BR12-winning_ticket.png', bbox_inches='tight')
ax.figure.savefig('e1_BE12-winning_ticket.png', bbox_inches='tight')



# Significance testing:  Paired significance tests.  each test compares
# baseline to each fixedfilter method.  Samples used for test are the test set
# dice of last 10 epochs of the 6 runs.
def yesno(pval, th=.1):
    return f'{"YES" if pval < th else "NO"} ({pval:0.5g})'
tbl = []
tblheader = ['Epoch', 'Method', '2-sided p-value', '1-sided p-value', 'is converged']  # Outperforms p-value', 'Underperforms p-value']
for se in range(0, 301, 1):
    bandwidth=10
    z = df.query(f'epoch>{se-bandwidth} and epoch <= {se+bandwidth}')['Dice']
    baseline = z.loc['Learned Baseline'].sort_index(level='epoch')
    for ffmethod in fixed_filter_methods + ['Learned Baseline']:
        is_converged = (0 if se<bandwidth or se > 301-bandwidth else (ks_2samp(
            df.query(f'epoch>{se} and epoch<{se+bandwidth}').loc[ffmethod]['Dice'].values,
            df.query(f'epoch>{se-bandwidth} and epoch<{se}').loc[ffmethod]['Dice'].values).pvalue))
        if ffmethod != 'Learned Baseline':
            is_different_pval = wilcoxon(
                z.loc[ffmethod].sort_index(level='epoch').values,
                baseline.values).pvalue
            is_worse_pval = wilcoxon(
                z.loc[ffmethod].sort_index(level='epoch').values,
                baseline.values, alternative='less').pvalue
        else:
            is_different_pval, is_worse_pval = 0,0
        #  is_better_pval = wilcoxon(
            #  z.loc[ffmethod].sort_index(level='epoch').values,
            #  baseline.values, alternative='greater').pvalue
        #  NOTE:  is_better_pval == 1-is_worse_pval
        #  tbl.append((se, ffmethod, yesno(is_different_pval), is_worse_pval, is_better_pval))
        tbl.append((se, ffmethod, is_different_pval, is_worse_pval, is_converged))
tbl = pd.DataFrame(tbl, columns=tblheader)

plt.ion()

###
## the really cool p-value plots.
###
#  z = tbl.pivot(
#    #'Underperforms p-value', 'Outperforms p-value'])\
#      'Epoch', 'Method', ['1-sided p-value'])\
#          .swaplevel(axis=1).sort_index(axis=1)
#  fig, axs = plt.subplots(1+len(fixed_filter_methods), 1, clear=True, figsize=(20,20))
#  #  fig.suptitle('Comparing Fixed Filter Methods to Baseline with Wilcoxon signed-rank Test')
#  for ax, col in zip(axs.ravel(), fixed_filter_methods):
#      #  ax.hlines([0.1, .9], 0, 300, linestyle='dashed', color='gray', linewidth=1, label='.05 confidence threshold')
#      z[col].plot(ax=ax)
#      ax.set_xlim(0, 300)
#      ax.fill_between(np.arange(1, 301), .90, 1, color='green', alpha=.2, label='Outperforms Baseline')
#      ax.fill_between(np.arange(1, 301), 0, .10, color='red', alpha=.2, label='Underperforms Baseline')
#      ax.vlines(214, 0, 1, linestyle='dotted', color='black', linewidth=1, label='Learned Baseline converged')
#      ax.set_title(f'Fixed Filter Method: {col}')
#      ax.legend()
#  ax = axs.ravel()[-1]
#  # tODO: uncomment
#  #  sns.lineplot(x='epoch', y='Dice', data=df.loc[['Learned Baseline', 'Psine']].reset_index(), ax=ax, ci=90)
#  ax.set_title('Learned Baseline, Test Set Performance (6 runs, 90% confidence interval)')
#  ax.set_xlim(0, 300)
#  ax.set_ylim(.7, .8)
#  ax.vlines(214, .55, .8, linestyle='dotted', color='red', linewidth=1, label='Learned Baseline converged')
#  ax.legend()
#  #  90% of of repeated experiments would significantly outperform baseline.
#  #  90% of of repeated experiments would significantly underperform baseline.
#  fig.tight_layout()
#  fig.savefig('e1-sigtests.png', bbox_inches='tight')
#  plt.close(fig)

#  tbl.pivot_table(['Underperforms p-value', 'Outperforms p-value'], 'Epoch', 'Method').swaplevel(axis=1).sort_index(axis=1)


###
# main plot:  line plots (test set learning curve, comparing methods)
###
#  experiment_names = [x for x in df.groupby('run_id').groups.keys()
                    #  if x not in {'Learned Baseline', 'SVDsteering'}]
experiment_names = fixed_filter_methods  #['Ones', 'DCT2', 'Unchanged', 'GHaar', 'Psine']
N = len(experiment_names)
#  fig, axs = plt.subplots(N, 1, figsize=(15, 16), sharex=True, sharey=True,
                        #  subplot_kw=dict(ylim=(.65, .81)), clear=True)
#  ylim = (.65, .81)  # BE11 or BE10
ylim = (.77, .88)  # BE12 or BR12
fig, axs = plt.subplots(N, 1, figsize=(15, 16), sharex=True, sharey=True,
                        subplot_kw=dict(ylim=ylim), clear=True)
axs.ravel()[0].set_title('BBBC038v1 Results with U-NetD Architecture', fontsize=30)
for ax, run_id in zip(axs.ravel(), experiment_names):
    # add the background color, based on p-value
    z = tbl.pivot(
      #'Underperforms p-value', 'Outperforms p-value'])\
        'Epoch', 'Method', ['1-sided p-value'])\
            .swaplevel(axis=1).sort_index(axis=1)[run_id]
    for epoch in z.index.values:
        pvalue = z.loc[epoch].values
        assert len(pvalue) == 1
        # thresholded red, yellow, green at 95% significance
        pvalue = 1. if pvalue[0] > .999 else (0. if pvalue[0] < .001 else .5)
        ax.fill_betweenx([0,1], epoch, epoch+1, color=plt.cm.RdYlGn(pvalue), alpha=.1)
    # annotate whether initialization is steered
    if any(x in run_id for x in {'DCT2', 'Ones'}):
        txt = "Non-Steered Initialization" 
    else:
        txt = "Steered Initialization" 
    # for BE11 or BE10
    #  ax.annotate(txt, (110, .66), fontsize=24,  bbox=dict(facecolor='white', edgecolor='gray', alpha=.9))
    # for BE12
    ax.annotate(txt, (110, .78), fontsize=24,  bbox=dict(facecolor='white', edgecolor='gray', alpha=.9))
    # add main result, a lineplot (slow to compute)
    data = df.loc[['Learned Baseline', run_id], 'Dice'].reset_index()
    data['Test Set Dice'] = data['Dice']
    data['Epoch'] = data['epoch']
    data['Experiment'] = data['run_id']
    sns.lineplot(
        x='Epoch', y='Test Set Dice', hue='Experiment',
        palette=sns.color_palette([
            (1.0, 0.4980392156862745, 0.054901960784313725),  # orange
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # blue
        ]), data=data, ax=ax)
    ax.legend(loc='lower right')

    # --> legend with background color on it
    #  import matplotlib.patches as mpatches
    #  handles, labels = ax.get_legend_handles_labels()
    #  handles.append(mpatches.Patch(color='red', alpha=.3, label='Underperforms Baseline'))
    #  handles.append(mpatches.Patch(color='green', alpha=.3, label='Outperforms Baseline'))
    #  ax.legend(handles=handles, loc='lower right')
    #  fig.suptitle("Microscopy Results")
             #  "\n1. Initialization should be steered.                    "
             #  "\n2. Fixed Filters perform better early in training.      "
             #  "\nEach line represents test set Dice on 6 trained models. ",
             #  horizontalalignment='left', x=.2)
#  fig.savefig('./e1-BE10-results.png', bbox_inches='tight')
#  fig.savefig('./e1-BE11-results.png', bbox_inches='tight')
#  fig.savefig('./e1-BR12-results.png', bbox_inches='tight')
fig.savefig('./e1-BE12-results.png', bbox_inches='tight')


#  # TODO: the order of experiment_names.


#  #  # 2. box plot of perf for each run (not used)
#  #  sort_order = df.rolling(10).mean().groupby(['run_id', 'filename'])['Dice'].median()\
#  #          .groupby('run_id').max().sort_values().index.tolist()
#  #  # --> tally the filenames so box plot categories don't look funky
#  #  def f():
#  #      dct = {}
#  #      # tally filenames
#  #      for x in df.index:
#  #          run_id, filename = x[:2]
#  #          dct.setdefault(run_id, {})
#  #          dct[run_id].setdefault(filename, len(dct[run_id]))
#  #          tally = dct[run_id][filename]
#  #          #  yield x, tally
#  #          yield tally
#  #  df['run_idx'] = list(f())
#  #  # --> all runs
#  #  fig, ax = plt.subplots(1,1,figsize=(15,15))
#  #  N = len(df.reset_index()['run_id'].unique())-.5
#  #  g = sns.boxplot(
#  #      x='run_id', y='Dice', hue='run_idx', whis=float('inf'),
#  #      data=df.reset_index(), order=sort_order, ax=ax) ; g.legend_.remove()
#  #  lines = [
#  #      ax.hlines(
#  #          df.xs('Learned Baseline').groupby('filename')['Dice'].max().mean(),
#  #          -.5, N, color='gray', linestyle='dashed', alpha=.5, linewidth=1),
#  #      ax.hlines(
#  #          df.xs('Learned Baseline').groupby('filename')['Dice'].max().max(),
#  #          -.5, N, color='gray', linestyle='dotted', alpha=.5, linewidth=1),
#  #      ax.hlines(
#  #          df.xs('Learned Baseline').groupby('filename')['Dice'].max().min(),
#  #          -.5, N, color='gray', linestyle='dotted', alpha=.5, linewidth=1)
#  #  ]
#  #  l = ax.legend(lines, [
#  #      'Baseline, mean of max scores',
#  #      'Baseline, max of max scores',
#  #      'Baseline, min of max scores',
#  #  ])
#  #  #  fig.autofmt_xdate(rotation=20)
#  #  ax.set_title("Distribution of Test Set Dice Score Across Epochs")
#  #  ax.set_xlabel('100% Fixed Spatial Filter Experiments')
#  #  ax.set_ylim(.5, .9)
#  #  # --> add the horizontal line to compare against baseline
#  #  fig.savefig('e1-box_plot_all_runs.png', bbox_inches='tight')

#  #  plt.show()
