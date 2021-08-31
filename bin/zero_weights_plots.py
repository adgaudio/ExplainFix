import pandas as pd
import glob
import re
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 18,
                     'legend.title_fontsize': 16,
                     'legend.fontsize': 15,
                     'axes.labelsize': 16,
                     'lines.markersize': 18,
                     })


import argparse as ap

def arg_parser():
    p = ap.ArgumentParser()
    p.add_argument('--old-prune-roc-vs-weights', action='store_true', help='DEPRECATED pruning scatterplot roc auc vs num weights')
    p.add_argument('--old-prune-timing', action='store_true', help='DEPRECATED plot timing results of pruning.')
    p.add_argument('--exp1', action='store_true', help='sanity check saliency metric, for appendix')
    p.add_argument('--exp2', action='store_true', help='zeroed and fixed weight models, unnecessary for inference and for training plots')
    p.add_argument('--exp3', action='store_true', help='Pruning scatter plot results')
    p.add_argument('--exp4', action='store_true', help='Timing of pruning and fixed')
    p.add_argument('--exp5', action='store_true', help='Table: Comparison of % zeroed and % pruned')
    return p


#  if NS.exp1 or NS.exp2:
def get_df(fps, parse_fp, names):
    tmp = [(parse_fp(fp), pd.read_csv(fp)) for fp in fps]
    assert len(dict(tmp)) == len(tmp), 'sanity check: no overlapping keys'
    df = pd.concat(dict(tmp), names=names).reset_index()
    df['Model'] = df['Model'].str.capitalize()
    #  if 'Is Pretrained' in df.columns:
        #  df['Is Pretrained'] = df['Is Pretrained'].str.capitalize()
    return df

if __name__ == "__main__":
    #  fps_e4 = glob.glob('zero_weights/zero_weights_v2_pct*).csv')
    #  df4 = get_df(
        #  fps_e4,
        #  #  lambda fp: re.match(r'zero_.*pct(\d+.?\d*)_(.*?pruned|)_?traineval_(baseline|Random_Fixed|FixedBaseline|GuidedSteer|Unchanged)_\((.*)\).*.csv', fp).groups(),
        #  lambda fp: re.search(r'/zero_.*pct(\d+.?\d*)_(.*?pruned|)_traineval_(baseline|Random_Fixed|FixedBaseline|GuidedSteer|Unchanged|DCT2|GHaar|Psine)_\((.*)\).*.csv', fp).groups(),
        #  ['pct', 'Pruned', 'Mode', 'Model'])
    #  import sys ; sys.exit()


    NS = arg_parser().parse_args()

    if (all(x in {False, None} for x in NS.__dict__.values())):
        print("DOING NOTHING!  Pass commandline args")
        print(arg_parser().print_help())
        import sys ; sys.exit()

    if NS.exp1:
        fps_e1 = glob.glob('zero_weights/zero_weights_base*.csv')
        fps_e1 += glob.glob('zero_weights/zero_weights_1_base*.csv')
        fps_e1 += glob.glob('zero_weights/zero_weights_2_base*.csv')
        fps_e1 += glob.glob('zero_weights/zero_weights_3_base*.csv')
        fps_e1 += glob.glob('zero_weights/zero_weights_4_base*.csv')
        fps_e1r = glob.glob('zero_weights/zero_weights_REVERSE_base*.csv')
        fps_e1r += glob.glob('zero_weights/zero_weights_REVERSE_1_base*.csv')
        fps_e1r += glob.glob('zero_weights/zero_weights_REVERSE_2_base*.csv')
        fps_e1r += glob.glob('zero_weights/zero_weights_REVERSE_3_base*.csv')
        fps_e1r += glob.glob('zero_weights/zero_weights_REVERSE_4_base*.csv')
        df1 = get_df(
            fps_e1,
            (lambda fp: re.match(r'zero.*?(\d?)_?(baseline|GuidedSteer|Unchanged)_\((.*)_(pretrained|fromscratch)\).csv', fp).groups()),
            ['iter', 'Mode', 'Model', 'Is Pretrained'])
        df1r = get_df(
            fps_e1r,
            (lambda fp: re.match(r'zero.*?(\d?)_?(baseline|GuidedSteer|Unchanged)_\((.*)_(pretrained|fromscratch)\).csv', fp).groups()),
            ['iter', 'Mode', 'Model', 'Is Pretrained'])
        df1[ 'Percent Least Salient Filters Zeroed Out'] = df1['Percentage Filters Zeroed Out']
        df1r['Percent Most Salient Filters Zeroed Out'] = df1r['Percentage Filters Zeroed Out']
        del df1['Percentage Filters Zeroed Out'], df1r['Percentage Filters Zeroed Out']

        #  # experiment 1: progressively removing least salient filters
        fig1, ax1 = plt.subplots(figsize=(12,4))
        sns.lineplot(
            data=df1,
            x='Percent Least Salient Filters Zeroed Out',
            y='Test Set Mean ROC AUC',
            hue='Model', style='Is Pretrained', ax=ax1)
        ax1.legend(loc='lower left', ncol=2)
        ax1.set_ylim(.4, 1)
        ax1.set_xscale('log');  ax1.set_xlabel(ax1.get_xlabel() + ' (Log Scale)')
        fig1.savefig('zero_weights_eval_logscale.png', bbox_inches='tight')
        #  fig1.savefig('zero_weights_eval.png', bbox_inches='tight')
        del fig1, ax1

        #  # --> experiment 1, verifying the saliency metric
        fig1r, ax1r = plt.subplots(figsize=(12,4))
        sns.lineplot(
            data=df1r.reset_index(),
            x='Percent Most Salient Filters Zeroed Out',
            y='Test Set Mean ROC AUC',
            hue='Model', style='Is Pretrained', ax=ax1r)
        ax1r.legend(loc='upper right', ncol=2)
        ax1r.set_ylim(.4, 1)
        ax1r.set_xscale('log');  ax1r.set_xlabel(ax1r.get_xlabel() + ' (Log Scale)')
        fig1r.savefig('zero_weights_eval_REVERSE_logscale.png', bbox_inches='tight')
        #  fig1r.savefig('zero_weights_eval_REVERSE.png', bbox_inches='tight')

    if NS.exp2:
        #  fps_e2 = glob.glob('zero_weights/zero_weights_pct*).csv')
        fps_e2 = glob.glob('zero_weights/exp2/*zero_weights_v2_pct*).csv')

        df2 = get_df(
            fps_e2,
            #  lambda fp: re.match(r'zero_.*pct(\d+.?\d*)_(.*?pruned|)_?traineval_(baseline|Random_Fixed|FixedBaseline|GuidedSteer|Unchanged)_\((.*)\).*.csv', fp).groups(),
            lambda fp: re.search(r'/(\d)_zero_.*pct(\d+.?\d*)_(.*?pruned|)_traineval_(baseline|Random_Fixed|FixedBaseline|GuidedSteer|Unchanged)_\((.*)\).*.csv', fp).groups(),
            ['iter', 'pct', 'Pruned', 'Mode', 'Model'])
        # use the Percentage filters number from filename, as the one in csv file (correctly) sets
        # first epoch to 0 pct filters removed.  But we want to use the percent to
        # identify the whole experiment including first epoch.
        df2['Percent Filters Zeroed Out'] = df2['pct'].astype('float').astype('category')
        del df2['pct']

        # experiment 2, unnecessary for inference.  spatial filters unnecessary for inference after fine-tuning non-spatial
        plt.rcParams.update({'font.size': 14})
        P = 'Percentage Filters Zeroed Out'
        df2[P] = df2[P].astype('category')
        df2a = df2.copy()[df2['Model'].str.contains('pretrained')].query('Pruned == "notpruned"')
        assert (df2a.groupby(['Model', 'iter']).count() == 162).values.all(), 'sanity check: missing data'
        mask = df2a[P] != 0
        df2a['Experiment'] = ''
        df2a.loc[~mask, 'Experiment'] = df2a.loc[~mask,'Model'].str.split('_', expand=True)[0] + ' Baseline: Fully Learned and Pre-trained'
        df2a.loc[mask, 'Experiment'] = '100% Spatially Fixed, Pre-trained, ' + df2a.loc[mask,P].astype('int').astype('str') + '% Spatial Kernels Zeroed ' + df2a.loc[mask,'Model'].str.split('_', expand=True)[0]
        #  df2a['Mode'].str.replace('_', ' ').replace('Unchanged', 'Spatially Fixed then Fine-Tuned') + ' ' + df2a['Model'].str.split('_', expand=True)[0] + ', ' + df2a[P].astype('str') + '% Spatial Kernels Zeroed, then\nNon-Spatial Weights Re-trained'
        df2a.rename({'Experiment': 'Experiment: Unnecessary for Inference'}, axis=1, inplace=True)
        #  df2a['Experiment'][(df2a[P] == 0) & (df2a['Mode'] == 'baseline')] = df2a['Experiment']
        # --> fig 1: show that in fine-tuned models, up to 99\% are unnecessary for inference
        grps = df2a[df2a['Model'].str.endswith('pretrained')].groupby('Model')
        fig2, axs2 = plt.subplots(grps.ngroups, 1, figsize=(12,8), sharey=True, sharex=True)
        for ax, (model, grp) in zip(axs2.ravel(), grps):
            grp = grp.sort_values([P, 'Epoch']).copy()
            #  # add baseline horizontal line
            _min_epoch, _max_epoch = grp['Epoch'].min(), grp['Epoch'].max()
            #  _tmp = grp.query(f'Mode=="baseline" and `{P}`==0')
            #  _tmp = _tmp.query(f'Epoch == {_tmp["Epoch"].min()}')['Test Set Mean ROC AUC']
            #  assert len(_tmp) == 6, 'sanity check: baseline repeated six times'
            #  _tmp = _tmp.unique()
            #  assert len(_tmp) == 1, 'sanity check: all baselines start from same point'  # actually incorrect assert
            #  ax.hlines(.863, _min_epoch, _max_epoch, linestyle='--', color='gray', alpha=.4, label='.863 Baseline')
            # Version 2 subplots
            sns.lineplot(
                hue='Experiment: Unnecessary for Inference',
                #  style='Model',
                y='Test Set Mean ROC AUC', x='Epoch',
                data=grp,
                #  legend='full',  # magically converts the hue to categorical not numeric values
                ax=ax, palette=sns.color_palette(['gray', 'green']))
            # Version 1 subplots
            #  # the line plots, one line per experiment
            #  sns.lineplot(
            #      hue='Percentage Filters Zeroed Out',
            #      style='Model',
            #      y='Test Set Mean ROC AUC', x='Epoch',
            #      data=grp, estimator=None,
            #      #  legend='full',  # magically converts the hue to categorical not numeric values
            #      ax=ax, palette='icefire')
            # prettify the legend.
            #  _a, _b = ax.get_legend_handles_labels()
            #  _b[1:1] = [f'({model.replace("_", " ").capitalize()})']
            #  _a[1:1] = [plt.Line2D([0],[0],color="none")]
            #  ax.legend(_a[:-2], _b[:-2], ncol=5, loc='lower right')
            #  ax.set_ylim(.75,.9)
        for ax in axs2.ravel()[[0,2]]:
            ax.set_ylabel(None)
        for ax in axs2.ravel():
            ax.set_ylim(0,1)
        fig2.tight_layout()
        #  fig2.subplots_adjust(left=.0)
        fig2.savefig('zero_weights_traineval_unnecessary_for_inference.png', bbox_inches='tight')

        # experiment 2: unnecessary for training.
        # --> fig 2: show that in trained from scratch models, up to 95\% are unnecessary for training and inference
        # --> fig 2: experiment 2, spatial filters unnecessary for training and inference
        df2b = df2.copy()[df2['Model'].str.contains('fromscratch')].query('Pruned == "notpruned"')
        assert (df2b.groupby(['Model', 'iter']).count() == 162).values.all(), 'sanity check: missing data'
        df2b['Experiment'] = ''
        mask = df2b[P] != 0
        df2b.loc[~mask, 'Experiment'] = df2b.loc[~mask,'Model'].str.split('_', expand=True)[0] + ' Baseline, Fully Learned'
        df2b.loc[mask, 'Experiment'] = df2b.loc[mask,'Model'].str.split('_', expand=True)[0] + ', 100% Spatially Fixed with Random Values and ' + df2b.loc[mask,P].astype('int').astype('str') + '% Spatial Kernels Zeroed'
        df2b.rename({'Experiment': 'Experiment: Unnecessary for Training (and Inference)'},
                   axis=1, inplace=True)
        grps = df2b[df2b['Model'].str.endswith('fromscratch')].groupby('Model')
        fig2b, axs2b = plt.subplots(grps.ngroups, 1, figsize=(12,8), sharey=False, sharex=True, squeeze=False)
        for ax, (model, grp) in zip(axs2b.ravel(), grps):
            grp = grp.sort_values([P, 'Epoch']).copy()
            # note: these were actually trained starting from epoch 1, not 40.  I
            # didn't correctly reset the epoch number in zero_weights.py, so I'll do it
            # here...
            grp['Epoch'] -= 39  # should start from epoch 1 not epoch 40.  
            #  grp = grp.query('Epoch<=50')
            #  # add baseline horizontal line
            _min_epoch, _max_epoch = grp['Epoch'].min(), grp['Epoch'].max()
            #  _tmp = grp.query(f'Mode=="baseline" and `{P}`==0')
            #  _tmp = _tmp.query(f'Epoch == {_tmp["Epoch"].min()}')['Test Set Mean ROC AUC']
            #  assert len(_tmp) == 6, 'sanity check: baseline repeated six times'
            #  _tmp = _tmp.mean()
            #  ax.hlines(.863, _min_epoch, _max_epoch, linestyle='--', color='gray', alpha=.4)
            # Version 2 subplots
            sns.lineplot(
                hue='Experiment: Unnecessary for Training (and Inference)',
                y='Test Set Mean ROC AUC', x='Epoch',
                data=grp,
                ax=ax, palette=sns.color_palette(['gray', 'green']))
            #  ax.set_ylim(.7, .9)
            ax.set_ylim(0,1)
        for ax in axs2b.ravel()[[0,2]]:
            ax.set_ylabel(None)
        fig2b.tight_layout()
        #  fig2.subplots_adjust(left=.0)
        fig2b.savefig('zero_weights_traineval_unnecessary_for_training_and_inference.png', bbox_inches='tight')

        #  import IPython ; IPython.embed() ; import sys ; sys.exit()

        ######
        # version 2 of exp2 plots
        ######
        mask = df2a[P] != 0
        df2a.loc[~mask, 'Experiment'] = ' Baseline: Fully Learned and Pre-trained'
        df2a.loc[mask, 'Experiment'] = '100\% Spatially Fixed and Pre-trained'
        mask = df2b[P] != 0
        df2b.loc[~mask, 'Experiment'] = ' Baseline: Fully Learned'
        df2b.loc[mask, 'Experiment'] = '100\% Spatially Fixed with Random Values'
        df2a.to_csv('tmpa.csv')
        df2b.to_csv('tmpb.csv')


        x='Percent Filters Zeroed Out'
        y='Test Set Mean ROC AUC'
        hue='Architecture'
        plt.rcParams.update({ "text.usetex": True, })
        for dft, experiment in [(df2a.copy(), 'Unnecessary for Inference'),
                               (df2b.copy(), 'Unnecessary for Training')]:
            dft = dft.sort_values(
                [P], ascending=True, kind='stable')
            plt.close('all')
            style=experiment
            dft[experiment] = dft['Experiment']  # overrides previously used column!
            dft['Architecture'] = dft['Model']\
                    .str.replace('_(fromscratch|pretrained)', '')\
                    .str.replace('Densenet', 'DenseNet121')\
                    .str.replace('Resnet', 'ResNet50')\
                    .str.replace('Efficientnet', 'EfficientNet-b0')
            g = sns.relplot(
                row='Architecture', legend='brief',
                data=dft.query('Epoch >60 and Epoch < 80').sort_values(style, ascending=False),
                x=x,y=y,hue=style, height=1.5, aspect=2,
                markers=['.', '.'],
            )
            for ax in g.fig.axes:
                ax.hlines(.86, 0, 100, label=None, color='gray', alpha=.5, linestyle='-.')
            # legend on top
            handles, labels = g.axes.flat[0].get_legend_handles_labels()
            g._legend.remove()
            g.fig.legend(
                reversed(handles), reversed(labels), ncol=1,
                loc='upper right', bbox_to_anchor=(.63, 0), frameon=True, fancybox=True)
            g.set(xlim=(-5,105), ylim=(0,1))
            g.axes.flat[0].set_ylabel('')
            g.axes.flat[-1].set_ylabel('')
            for ax, (w1,w2) in zip(g.axes.ravel(), [(0,85), (0,95), (0,99), ]):
                #  ax.arrow(0, .60, 1, 0)
                ax.annotate(
                    "", xytext=(w1-.5, .60), xy=(w2+.5, 0.60),
                    ha='right', va='center', arrowprops=dict(
                        arrowstyle="|-|"))
                pct = dft.loc[dft['Architecture'] == ax.get_title().split()[-1], x].astype('float').max()
                #  ax.text(
                    #  10, .60, r'\bf{' + f'{pct:0.0f}\% {experiment}' + r'.}',
                    #  ha='left', va='top', fontsize=16)
                ax.text(
                    33, .50, r'\bf{' + "100x Fewer" + r'.}',
                    ha='left', va='top', fontsize=20)
            g.fig.subplots_adjust(top=1, bottom=.8)
            #  g.fig.suptitle(experiment, fontsize=25)
            #  g.fig.suptitle(experiment)
            g.fig.tight_layout()
            #  g.fig.subplots_adjust(bottom=0.4, top=1.4)
            if 'Training' in experiment:
                g.fig.savefig('zero_weights_traineval_unnecessary_for_training_and_inference_v2.png', bbox_inches='tight', pad_inches=.02)
            else:
                g.fig.savefig('zero_weights_traineval_unnecessary_for_inference_v2.png', bbox_inches='tight', pad_inches=.02)

            # and just a single image version for the fig 1, from christos.
            [ax.remove() for ax in g.fig.axes[:-1]]
            z = g.fig.axes[-1]
            g.fig.set_figheight(g.fig.get_figheight()/2)
            z.change_geometry(1,1,1)
            #  z.set_position([0,0,1,1])
            z.set_ylabel('Test Set\nMean\nROC AUC')
            z.set_xlabel('Percent Spatial Filters Zeroed Out')
            #  g.fig.tight_layout()
            g.fig.subplots_adjust(bottom=.2)
            if 'Training' in experiment:
                pass
                #  g.fig.savefig('zero_weights_traineval_unnecessary_for_training_and_inference_v3.png', bbox_inches='tight', pad_inches=.02)
            else:
                #  z.text(
                    #  10, .40, r'\bf{' + f'90\% Unnecessary for Training' + r'.}',
                    #  ha='left', va='top', fontsize=16)
                g.fig.legends[0].remove()
                g.fig.legend(
                    reversed(handles), ['Baseline: Fully Learned', 'Ours: Spatially Fixed and Sparse'], ncol=2,
                    loc='upper left', bbox_to_anchor=(-.00, 0), frameon=True, fancybox=True)
                g.fig.savefig('zero_weights_traineval_unnecessary_for_inference_v3.png', bbox_inches='tight', pad_inches=.02)


    # deprecated
    if NS.old_prune_timing:
        # experiment 3: pruning the zero weights models
        df = pd.read_csv('zero_weights/timing_experiment_results.csv')
        df = df.groupby(['Experiment', 'model_', 'Is Pruned']).mean()
        df = df.reset_index()
        df2 = df.copy()
        def get_spe_pct_savings(grp):
            baseline_spe = grp.query('Experiment == "Baseline"')['Spatially Fixed'].values
            return (baseline_spe - grp['Spatially Fixed']) / baseline_spe * 100
        def get_baseline_spe(grp):
            baseline_spe = grp.query('Experiment == "Baseline"')['Spatially Fixed'].values[0]
            return grp['Spatially Fixed'] * 0 + baseline_spe
        def get_pruning_spe(grp):
            pruned_spe = grp.query('`Is Pruned` == "Pruned"')['Spatially Fixed'].values[0]
            return grp['Spatially Fixed'] * 0 + pruned_spe
        df['% savings'] = df.groupby(['model_']).apply(get_spe_pct_savings).reset_index('model_', drop=True)
        df['Learned Baseline'] = (  # seconds per epoch
                df.groupby(['model_']).apply(get_baseline_spe).reset_index('model_', drop=True) )
        df['Model'] = df['model_'].str.split(' ', expand=True)[0].str.title()
        df = df.query('Experiment != "Baseline"')
        df['After Pruning'] = (  # seconds per epoch
                df.groupby(['model_']).apply(get_pruning_spe).reset_index('model_', drop=True) )
        df = df.query('`Is Pruned` == "Not Pruned"')
        df = df.sort_values(['Experiment', 'model_'])
        from zero_weights_prune import cfg_files
        df['Pruning Amount'] = [
            f'{x:g}% Zeroed, {y:g}% Pruned' for x,y in 
            # percent zeroed numbers
            (pd.read_csv('./zero_weights/pruned_model_amounts.csv')
             .set_index('Unnamed: 0')
             .reindex((df['Experiment'] + ' (' + df['model_'].str.lower() + ')').values
                      )[['Pct Spatial Weights Zeroed', 'Pct Spatial Weights Pruned']].round().astype('int').values)
                      ]
        #  df['Experiment'].replace({'Random fixed': 'Train non-spatial weights of Fixed, Pruned Random Network', 'Unchanged': 'Unnecessary for Inference'}, inplace=True)
        fig3, ax = plt.subplots(1, 1, figsize=(12,4))
        df.rename({'Spatially Fixed': '100% Spatially Fixed'}, axis=1, inplace=True)
        df.set_index(['Model', 'Pruning Amount'])[['Learned Baseline', '100% Spatially Fixed', 'After Pruning']].T.plot(style='-o', ax=ax)
        ax.set_ylabel('Seconds to Train One Epoch')
        ax.set_xticklabels(ax.get_xticklabels(), fontdict={'fontsize': 18})
        title_proxy = Rectangle((0,0), 0, 0, color='none')
        h,l = ax.get_legend_handles_labels()
        ax.legend([title_proxy, *h[3:], title_proxy, title_proxy, *h[:3]], 
                  ["Unnecessary for Inference Models", *l[3:], None, "Unnecessary for Training Models", *l[:3]],
                   bbox_to_anchor=(1,.5), loc="center left")
        fig3.tight_layout()
        plt.show()
        ax.figure.savefig('zero_weights_timing_analysis.png', bbox_inches='tight')

    # deprecated
    if NS.old_prune_roc_vs_weights:
        import scipy.stats as st
        fully_learned_baselines = {f'{mdl} {fromscratch}': check_output(
            f'cat results/C{{A,}}8-*{mdl.lower()}*{fromscratch.lower().replace("-","")}*unmodified_baseline*/eval_CheXpert_*d.csv|grep test_roc_auc_MEAN | cut -d, -f2', shell=True)
            for mdl in {'DenseNet', 'ResNet', 'EfficientNet'} for fromscratch in {'From-Scratch', 'Pre-Trained'}}
        fully_learned_baselines = {
            mdl: np.array([float(x) for x in out.decode().strip().split('\n')])
            for mdl, out in fully_learned_baselines.items()}
        fully_learned_baselines = pd.DataFrame(fully_learned_baselines)  # will fail if missing data
        f = fully_learned_baselines

        # Pruning scatter plot:  results of the zero_weights_prune.py
        # get baselines:
        df = pd.read_csv('./zero_weights/pruned_model_amounts_v2.csv', header=[0,1], index_col=0)
        # TODO: add the data for training seconds per epoch
        df.loc[df[('Other', 'Is Baseline?')] == True, 'After Pruning'] =  0
        fig, ax = plt.subplots(figsize=(12,4))
        kws = dict(
            x='Num Weights', y='Mean ROC AUC', hue='Model',
            data=df.stack('Pruning').reset_index(), ax=ax
        )
        ax = sns.lineplot(**kws, legend=False)
        ax = sns.scatterplot(**kws, style='Pruning')
        ax.legend(
           bbox_to_anchor=(1,.5), loc="center left")
        fig.savefig('pruning_rocauc_vs_numweights.png', bbox_inches='tight')

    if NS.exp3:
        from subprocess import check_output

        # get predictive performance results for fully learned baselines
        fully_learned_baselines = {
            (mdl, 'ImageNet' if is_pretrained == 'pretrained' else 'KUniform',): check_output(
            f'cat results/C{{A,}}8-*{mdl.lower()}:5:{is_pretrained}*unmodified_baseline*/eval_CheXpert_*d.csv|grep test_roc_auc_MEAN | cut -d, -f2', shell=True)
            for mdl in {'DenseNet121', 'ResNet50', 'EfficientNet-b0'} for is_pretrained in {'pretrained'}}
        fully_learned_baselines = {
            (f'Baseline: {mdl}, Learned Weights', mdl, pret): v
            for (mdl, pret), v in fully_learned_baselines.items()}
        fully_learned_baselines = {
            mdl: np.array([float(x) for x in out.decode().strip().split('\n')])
            for mdl, out in fully_learned_baselines.items()}
        fully_learned_baselines = pd.DataFrame(fully_learned_baselines)  # will fail if missing data
        f = fully_learned_baselines.unstack()
        f.name = 'Unpruned Baselines'
        f.index.names = ['Experiment', 'Model', 'Method', 'iter']
        # get data for pruned models
        version = 6
        fps = []
        for method in ['Pruned_Baseline', 'ImageNet']:
            fps.extend(glob.glob(
                f'zero_weights/*_zero_weights_v{version}_pct*_{method}_*).csv'))
        if version in [5,6]:
            parse_fp = lambda fp: re.match(
                r'.*?(\d_|)zero_weights_v[56]_pct(.*?)_pruned_traineval_(ImageNet|Pruned_Baseline)_\((.*?)_(pretrained)\).csv', fp).groups()
        else:
            raise NotImplementedError('how to parse the data filenames')
        #  fps = [x for x in fps if 'Pruned_Baseline' not in x]
        df = pd.concat({parse_fp(fp): pd.read_csv(fp) for fp in fps})
        df.index.names = ['iter', 'Pct Zeroed', 'Method', 'Model', 'is_pretrained', 'junk']
        df = df.reset_index().drop('junk', axis=1)
        df['Model'].replace({'resnet': 'ResNet50',
                             'densenet': 'DenseNet121',
                             'efficientnet': 'EfficientNet-b0'}, inplace=True)
        df['Method'].replace({'Pruned_Baseline': 'Pruned Baseline'}, inplace=True)
        df['Experiment'] = 'Pruned '
        mask = (df['Method'] != 'Pruned Baseline')
        df.loc[mask, 'Experiment'] += df.loc[mask, 'Model'] + ', Fixed Weights'# + ' ' + df.loc[mask, 'Pct Zeroed']
        df.loc[~mask, 'Experiment'] += df.loc[~mask, 'Model'] + ', Learned Weights'# + ' ' + df.loc[~mask, 'Pct Zeroed']
        df['iter'] = df['iter'].str.replace('_', '').astype('int') - 1
        # TODO: REMOVE.  don't use this
        #  plt.figure()
        #  df[['Experiment', 'Test Set Mean ROC AUC', 'Epoch']].pivot_table('Test Set Mean ROC AUC', 'Epoch', 'Experiment').plot()
        #  plt.show()
        #  import IPython ; IPython.embed() ; import sys ; sys.exit()
        tdata = pd.concat([
            # could show better perf of show more epochs, but I decided it might not be a fair test to do that, so stick with 40 epochs like on baselines.
            # doing exactly what is shown for baselines.
            df.query('Epoch >= 40 and Epoch <= 40 and Model == "ResNet50"'),
            df.query('Epoch >= 40 and Epoch <= 40 and Model == "DenseNet121"'),
        ]) .groupby(['Experiment', 'Model', 'Method', 'iter', 'Pct Zeroed']).agg(
            {'Test Set Mean ROC AUC': 'mean',
             'Num Weights After Pruning': pd.Series.mode,
             'Num Weights Before Pruning': pd.Series.mode,
             'Pct Weights Pruned': pd.Series.mode,})
        # join the two data sources to get the num weights for every unpruned baseline model
        df = tdata.reset_index('Pct Zeroed').join(f, how='outer').reset_index()
        df['Num Weights Before Pruning'] = df.groupby('Model')['Num Weights Before Pruning'].transform('first')
        df.loc[df['Experiment'].str.contains('Baseline:'), 'Pct Weights Pruned'] = 0
        df = pd.DataFrame(np.vstack([
            df[['Experiment', 'Model', 'Method', 'iter', 'Test Set Mean ROC AUC', 'Num Weights After Pruning', 'Pct Weights Pruned']].values,
            df[['Experiment', 'Model', 'Method', 'iter', 'Unpruned Baselines', 'Num Weights Before Pruning', 'Pct Weights Pruned']].values,
        ]), columns=['Experiment', 'Model', 'Method', 'iter', 'Test Set Mean ROC AUC', 'Num Weights', 'Pct Weights Pruned'])
        df['Is Pruned'] = ~df['Experiment'].str.contains('Baseline:')
        df['Is Learned'] = df['Experiment'].str.contains('Learned')
        df['groupby_pruned_and_learned'] = (df['Is Pruned'].values << 1) | df['Is Learned'].values
        df = df.sort_values(['Model', 'Is Learned', 'Is Pruned'], ascending=[True,True,False])
        # throw out EfficientNet-b0 because didn't implement ability to prune it
        # throw out any missing data so it doesn't appear in legend.
        df = df.query('Model != "EfficientNet-b0"')
        # ok rename the legend one last time... omg
        def rename_experiment(strng):
            strng = re.sub(r'Baseline: (.*?), Learned Weights',
                           r'Baseline: Fully Learned', strng)
            strng = re.sub(r'Pruned (.*?), Fixed Weights',
                           r'ExplainFix', strng)
            strng = re.sub(r'Pruned (.*?), Learned Weights',
                           r'ChannelPrune (Fully Learned)', strng)
            return strng
        df['Experiment'] = df['Experiment'].apply(rename_experiment)
        # prep the plot
        # Do some fancy stuff with colors to basically make a "saturation" option that sets unpruned baseline lighter and more gray, pruned baseline less gray, pruned fixed model strong colorful.
        #  grayish color: unpruned baseline
        #  darker gray color: pruned baseline
        #  strong color: pruned fixed model
        #  colors = {'ResNet50': sns.color_palette('crest', 5), #sns.cubehelix_palette(3, start=1, rot=.1),
                  #  'DenseNet121': sns.color_palette('flare', 5), #sns.cubehelix_palette(3, start=2, rot=.1),
                  #  'EfficientNet-b0': sns.cubehelix_palette(3, start=0, rot=.1),
        sns.set_context('talk')
        fig, ax = plt.subplots(figsize=(12, 4))
        # main scatter points
        sns.scatterplot(
            #  y='Test Set Mean ROC AUC', x='Num Weights',
            y='Test Set Mean ROC AUC', x='Pct Weights Pruned',
            hue='Model', markers=['+', '3', '_'], style='Experiment', s=200,
            data=df, ax=ax, alpha=.8)

        # horizontal dashed lines
        for name, g in f.groupby('Experiment'):
            if 'EfficientNet' in name:
                continue  # didn't prune it because of its implementation
            elif 'ResNet' in name:
                color = 'darkorange'
                label = None#'Baseline Mean ROC AUC'
            elif 'DenseNet' in name:
                color = 'steelblue'
                label = None
            else:
                raise NotImplementedError()
            ax.hlines(g.mean(), 0, 1, linewidth=1, linestyle='--', color=color, label=label)
        a,b = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        b.append('Baseline Mean ROC AUC')
        a.append(Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Baseline Mean ROC AUC'))
        b.insert(3, ' ')
        a.insert(3, Line2D([0], [0], color='none', label=' '))

        ax.legend(handles=a,
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.set_ylim(.8, .9)


        fig.tight_layout()
        fig.savefig('pruning_rocauc_vs_numweights.png', bbox_inches='tight', pad_inches=.05)
        # TODO: instead of "num weights" do xaxis like: pct weights, or (10x smaller, 5x smaller,...)

    if NS.exp4:
        plt.rcParams.update({ "text.usetex": True, })
        fps = glob.glob('zero_weights/timing*v6*).csv')

        parse_fp = lambda fp: re.match(
            r'.*?timingzero_weights_v6_pct(.*?)_(pruned|notpruned)_traineval_(baseline|ImageNet|Pruned_Baseline)_\((.*?)_(pretrained)\).csv', fp).groups()
        df = pd.concat({parse_fp(fp): pd.read_csv(fp) for fp in fps})
        df.index.set_names(['Pct Zeroed', 'Is Pruned', 'Method', 'Model', 'Is Pretrained', 'ignore'], inplace=True)
        df = df.reset_index(df.index.names[:-1])
        df['Model'].replace({'resnet': 'ResNet50', 'densenet': 'DenseNet121', 'efficientnet': 'EfficientNet-b0'}, inplace=True)
        df['Is Fixed'] = ~df['Method'].str.lower().str.contains('baseline')
        assert df.query('Epoch == 0 or Epoch == 40')['seconds_training_epoch'].isnull().all(), 'sanity check'
        df = df.query('Epoch != 0 and Epoch != 40')  # the first epoch reported in logs isn't trained.
        df['Experiment'] = ''
        mask = df['Is Fixed'] & (df['Is Pruned']=='pruned') & (df['Epoch'] > 0)
        df.loc[mask, 'Experiment'] = 'Fixed, ' + (df.loc[mask, 'Pct Weights Pruned']*100).astype('int').astype('str') + '% Pruned'
        mask = df['Is Fixed'] & (df['Is Pruned']=='notpruned')
        df.loc[mask, 'Experiment'] = 'Fixed, Not Pruned'
        mask = (~df['Is Fixed']) & (df['Is Pruned']=='notpruned')
        df.loc[mask, 'Experiment'] = 'Learned Baseline: Not Fixed, Not Pruned'
        assert df.loc[mask, 'Pct Weights Pruned'].isnull().all()
        df.loc[mask, 'Pct Weights Pruned'] = 0
        mask = (df['Is Fixed']) & (df['Is Pruned']=='notpruned')
        assert df.loc[mask, 'Pct Weights Pruned'].isnull().all()
        df.loc[mask, 'Pct Weights Pruned'] = 0
        df['Seconds per Epoch Training Non-Fixed Weights'] = df['seconds_training_epoch']
        df['Arch'] = df['Model']
        mask = (df['Is Fixed'])
        df.loc[mask, 'Model'] = "Ours: ExplainFix"
        mask2 = (df['Is Pruned'] == 'pruned')
        df.loc[(~mask) & (~mask2), 'Model'] = "Baseline: Fully Learned"
        df.loc[(~mask) & (mask2), 'Model'] = "ChannelPrune (Fully Learned)"

        #  df.groupby(['Arch', 'Pct Weights Pruned', 'Is Fixed'])['Seconds per Epoch Training Non-Fixed Weights'].mean()
        #  import IPython ; IPython.embed() ; import sys ; sys.exit()

        # For visual, Assign "pct weights pruned" to the group average value so can draw error bars.
        df['Pct Weights Pruned'] = df.groupby(['Arch', 'Percentage Filters Zeroed Out'])['Pct Weights Pruned'].transform(lambda x: (x.mean()*100).astype('int'))
        df = df[~df['Arch'].str.contains('EfficientNet')]
        df = df.sort_values(['Is Fixed', 'Is Pruned'])
        # num weights
        #  m = df['Is Pruned'] == 'pruned'
        #  _tmp = df.groupby(['Arch'])['Num Weights Before Pruning'].first()
        #  for arch in _tmp.index:
        #      df.loc[(~m) & (df['Arch'] == arch), 'Num Weights'] = _tmp.loc[arch]
        #  df.loc[m, 'Num Weights'] = df.loc[m, 'Num Weights After Pruning']
        #  assert df['Num Weights'].isnull().sum() == 0, 'bug'
        Z = df.groupby(['Arch', 'Pct Weights Pruned', 'Model'])['Seconds per Epoch Training Non-Fixed Weights'].mean()
        Z.to_csv('data_for_tbl_num_params.csv')
        print(Z.to_string())
        for yaxiscol in ['Seconds per Epoch Training Non-Fixed Weights', 'Test Set Mean ROC AUC']:
            fig, axs = plt.subplots(1,2, figsize=(12, 3), clear=True)
            figs, axs_simple = plt.subplots(1,2, figsize=(12, 3), clear=True)
            for ax,ax_simple, arch in zip(axs, axs_simple, df['Arch'].unique()):
                palette = sns.palettes.mpl_palette('tab10')
                palette[:2] = reversed(palette[:2])
                dfg = df.query(f'Arch == "{arch}"')

                dfg[['Pct Weights Pruned', yaxiscol, 'Model']].to_csv(f'{arch}.csv')
                if yaxiscol == 'Test Set Mean ROC AUC':
                    dfg = dfg.query('Epoch > 10')
                sns.pointplot(data=dfg,
                              x='Pct Weights Pruned',
                              y=yaxiscol,
                              hue='Model',
                              palette=palette,
                              ci=95, ax=ax)
                #  ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                #  ax.legend(loc='upper right', borderaxespad=0)
                ax.yaxis.set_major_locator(plt.MultipleLocator(10))
                ax.set_title(arch, fontsize=16)
                ax_simple.set_title(arch, fontsize=16)
                if yaxiscol == 'Test Set Mean ROC AUC':
                    ax_simple.set_ylim(0,1)

                if arch == 'ResNet50':
                    bestpct = 79
                    x_smaller = 5
                    x_faster = 17
                elif arch == 'DenseNet121':
                    bestpct = 74
                    x_smaller = 4
                    x_faster = 9
                else:
                    raise NotImplementedError()
                dfg_simple = pd.concat([
                    dfg.query('Model == "Baseline: Fully Learned"'),
                    dfg[(dfg['Is Pruned'] == 'pruned')
                        & dfg['Is Fixed'] & (dfg['Pct Weights Pruned'] == bestpct)]])
                sns.pointplot(data=dfg_simple,
                              x='Pct Weights Pruned',
                              y=yaxiscol,
                              hue='Model',
                              palette=reversed(sns.color_palette('tab10', 2)),
                              ci=95, ax=ax_simple,
                              s=200
                              )
                if yaxiscol == 'Seconds per Epoch Training Non-Fixed Weights':
                    txt = f'{x_faster}\% Faster'
                    ax_simple.text(
                        .85, 1.2, txt, rotation='vertical', transform=ax.transAxes,
                        ha='left', va='top', fontsize=20)
                    h1 = .94 if arch == 'ResNet50' else .90
                    h2 = 0.37 if arch == 'ResNet50' else .41
                    ax_simple.annotate(
                        "", xytext=(.78, h1), xy=(.78, h2),
                        xycoords='axes fraction',
                        ha='right', va='center', arrowprops=dict(
                            arrowstyle="|-|"))
                    # ... horizontal
                    h = .17 if arch == 'ResNet50' else .18
                    w2 = .695 if arch == 'ResNet50' else .690
                    w = .060
                    ax_simple.annotate(
                        "", xytext=(w, h), xy=(w2, h),
                        xycoords='axes fraction',
                        ha='right', va='center', arrowprops=dict(
                            arrowstyle="|-|"))
                    ax_simple.text(
                        w+.1, .5+h, f'{x_smaller}x Smaller',
                        rotation='horizontal', transform=ax.transAxes,
                        ha='left', va='top', fontsize=20)
                else:
                    h = .65 if arch == 'ResNet50' else .65
                    ax_simple.annotate(
                        "", xytext=(0.04, h), xy=(.963, h),
                        xycoords='axes fraction',
                        ha='right', va='center', arrowprops=dict(
                            arrowstyle="|-|"))
                    ax_simple.text(
                        .1, .75, f'{x_smaller}x Smaller\nMatching Accuracy',
                        rotation='horizontal', transform=ax.transAxes,
                        ha='left', va='top', fontsize=20)
                    #  ax_simple.set_xlim(.05,1.05)
                    ax_simple.hlines(.0, 0, 1, label=None, color='white', alpha=1, linestyle='-.')
                    ax_simple.hlines(.85, 0, 1, label=None, color='gray', linestyle='-.', linewidth=1, alpha=.5)

                # legend on bottom
                for figg, axx, ncol in [(fig, ax, 3), (figs, ax_simple, 2)]:
                    handles, labels = axx.get_legend_handles_labels()
                    axx.get_legend().remove()
                    figg.legend(
                        handles, labels, ncol=ncol,
                        loc='upper left', bbox_to_anchor=(0.01, 0), frameon=True, fancybox=True)
            for n, ax in enumerate(axs.ravel()):
                a,b = ax.get_xlim(); ax.set_xlim(a, b+(b-a)*.15)
                a,b = ax.get_ylim(); ax.set_ylim(a, b+(b-a)*.40)
                # horizontal arrows
                if n == 0:  # resnet
                    w1, w2 = -.02,3.02
                    h1, h2 = 243, 243
                    wt, ht = .5, .8
                    msgx, msgy = '7x Smaller', r'18\% Faster'
                else:  # densenet
                    w1, w2 = -.02,2.02
                    h1, h2 = 400,400
                    wt, ht = .5,.8
                    msgx, msgy = '4x Smaller', r'9\% Faster'
                ax.text(
                    wt, ht, msgx, rotation='horizontal',
                    transform=ax.transAxes, ha='center', va='top', fontsize=20)
                ax.annotate(
                    "", xytext=(w1, h1), xy=(w2, h2),
                    ha='right', va='center', arrowprops=dict(arrowstyle="|-|"))
                # vertical arrows
                if n == 0:  # resnet
                    w1, w2 = 3.40,3.40
                    h1, h2 = 188-1, 230+1  # weirdo positioning matplotlib!
                    wt, ht = .91, .5
                else:  # densenet
                    w1, w2 = 2.39,2.39
                    h1, h2 = 347-1,383+1
                    wt, ht = .91,.5
                ax.text(
                    wt, ht, msgy, rotation='vertical',
                    transform=ax.transAxes, ha='left', va='center', fontsize=20)
                ax.annotate(
                    "", xytext=(w1, h1), xy=(w2, h2),
                    ha='right', va='center', arrowprops=dict(arrowstyle="|-|"))

            axs[0].set_ylabel(None)
            axs[1].set_ylabel(None)
            axs_simple[0].set_ylabel(None)
            axs_simple[1].set_ylabel(None)
            if yaxiscol == 'Seconds per Epoch Training Non-Fixed Weights':
                ylabel = yaxiscol.replace('Epoch Training', 'Epoch\nTraining')
            else:
                ylabel = '\n'+yaxiscol
            fig.supylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
            figs.supylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
            fig.subplots_adjust(bottom=.25)
            figs.subplots_adjust(bottom=.25)

            if yaxiscol == 'Seconds per Epoch Training Non-Fixed Weights':
                figs.axes[0].set_xlim(-.1, 1.5)
                figs.axes[1].set_xlim(-.1, 1.5)
            a,b = figs.axes[0].get_ylim(); figs.axes[0].set_ylim(a-.6*(b-a),b+.1*(b-a))
            a,b = figs.axes[1].get_ylim(); figs.axes[1].set_ylim(a-.6*(b-a),b+.1*(b-a))
            #  figs.tight_layout()
            if yaxiscol == 'Seconds per Epoch Training Non-Fixed Weights':
                fig.savefig(f'zero_weights_timing_analysis.png', bbox_inches='tight', pad_inches=0.02)
                figs.savefig(f'zero_weights_timing_analysis_SIMPLE.png', bbox_inches='tight', pad_inches=0.02)
            else:
                fig.savefig('zero_weights_rocauc_vs_pctpruned.png', bbox_inches='tight', pad_inches=0.02)
                figs.savefig('zero_weights_rocauc_vs_pctpruned_SIMPLE.png', bbox_inches='tight', pad_inches=0.02)

        # histograms of the pct weights.
        _tmp = df.groupby(['Arch'])['Num Weights Before Pruning'].first()
        _tmp2 = {k: {'Pct Weights Pruned': [], 'Num Weights': []} for k in _tmp.index}
        for _, a, p in list(df[['Arch', 'Pct Weights Pruned']].drop_duplicates().sort_values('Pct Weights Pruned').itertuples()):
            _tmp2[a]['Num Weights'].append((1- p / 100) * _tmp.loc[a])
            _tmp2[a]['Pct Weights Pruned'].append(p)
        fig, axs = plt.subplots(1,2)
        _tmp2 = pd.concat({k:pd.DataFrame(v) for k, v in _tmp2.items()}, names=['Architecture', 'n']).reset_index()
        print(_tmp2)
        g = sns.FacetGrid(_tmp2, col='Architecture', sharex=False)
        g.map_dataframe(sns.barplot, y='Num Weights', x='Pct Weights Pruned')
        g.set_axis_labels('Pct Weights Pruned', 'Num Weights')
        g.set_titles(col_template="{col_name}")
        g.fig.savefig('zero_weights_timing_analysis_with_hist.png', bbox_inches='tight', pad_inches=0.02)

    if NS.exp5:
        fps = glob.glob('zero_weights/timing*v6*ImageNet*).csv')
        parse_fp = lambda fp: re.match(
            r'.*?timingzero_weights_v6_pct(.*?)_(pruned|notpruned)_traineval_(baseline|ImageNet|Pruned_Baseline)_\((.*?)_(pretrained)\).csv', fp).groups()
        df = pd.concat({parse_fp(fp): pd.read_csv(fp) for fp in fps})
        df.index.set_names(['Pct Zeroed', 'Is Pruned', 'Method', 'Model', 'Is Pretrained', 'ignore'], inplace=True)
        df = df.groupby(df.index.names[:-1])[['Pct Weights Pruned']].last().reset_index('Pct Zeroed').reset_index(['Is Pruned', 'Method', 'Is Pretrained'], drop=True).dropna()
        df = df.reset_index().replace({'resnet': 'ResNet50', 'densenet': 'DenseNet121'}).set_index('Model')
        print(df.to_latex())
