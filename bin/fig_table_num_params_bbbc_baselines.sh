
# get results
(
# echo "run_id,Total Parameters,Fixed Parameters,Free Parameters,Spatial Parameters,1x1 Parameters"
echo "run_id,Total Parameters,Num Fixed,Num Learned,Num Spatial,Num 1x1,Num Other"
# for model in \
  # 'densenet121:5:fromscratch' 'densenet169:5:fromscratch' 'densenet201:5:fromscratch' \
    # 'efficientnet-b0:5:fromscratch' 'efficientnet-b1:5:fromscratch' 'efficientnet-b2:5:fromscratch' 'efficientnet-b3:5:fromscratch' 'efficientnet-b4:5:fromscratch' \
    # 'resnet18:5:fromscratch' 'resnet50:5:fromscratch' 'resnet152:5:fromscratch' \
    # 'unetD_small' 'unet_pytorch_hub' 'deeplabv3plus_mobilenet'
for model in \
    'deeplabv3plus_resnet50' \
    'deeplabv3plus_mobilenet' \
    'unetD_small'
  do
  python bin/train.py --model $model --model-mode spatial_100%_unchanged --do-this numel |tail -n 1
  done ; wait) | tee | python <(cat <<EOF
import sys
import pandas as pd
df = pd.read_csv(sys.stdin, sep=',')
df.to_csv('tmp.csv', index=False)
EOF
)

python <(cat <<EOF
import pandas as pd
df = pd.read_csv('tmp.csv')
# df2 = pd.read_csv('tmp_seconds.csv')
assert df[['Num Spatial', 'Num 1x1', 'Num Other']].sum(1).equals(df['Total Parameters']), 'sanity check'
df['Model'] = df['run_id'].str.extract('(.*?)(:5:fromscratch|\-spatial_100|\-unmodif).*')[0]
# df2['Model'] = df2['run_id'].str.extract('C.?8-(.*?)(:5:fromscratch|\-spatial_100|\-unmodif).*')[0]
df = df.sort_values('Total Parameters').set_index('Model')
# df2 = df2.set_index('Model', append=False).reindex(df.index)
print(df[['Num Fixed', 'Num Learned']].to_string())
print()
# get the num of spatial params
tbl = df[['Num Spatial', 'Num 1x1', 'Num Other']].apply(lambda x: x.astype('str') + (' (' + (x/x.sum()*100).round(0).astype('int').astype('str') + '%)').str.rjust(6), axis=1)
tbl = df[['Total Parameters']].join(tbl[['Num Spatial']]).rename({'Total Parameters': 'Num Params', 'Num Spatial': 'Spatial Params(% total)'}, axis=1)

# get seconds per epoch from files
# tbl[(r'SPE (% savings)')] = df2['seconds_training_epoch']
with open('./tbl_num_params.tex', 'w') as fout:
  fout.write(tbl.to_latex())
print(tbl.to_string())
print('OUTPUT: tbl_num_params.tex')
# .plot.barh(stacked=True)
# print(df.to_string())
# print(df.to_latex())
EOF
)
