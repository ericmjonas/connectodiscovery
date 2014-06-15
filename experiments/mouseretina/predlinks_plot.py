import numpy as np
import cPickle as pickle
import pandas
from matplotlib import pylab


THOLD_IDX = 1


clust_df = pickle.load(open("cluster_metrics.pickle"))['clust_df']
clust_df = clust_df[ clust_df['filename'].str.contains("debug") == False]

clust_df = clust_df[clust_df['filename'].str.contains("retina.%d" % THOLD_IDX)]

def f(x):
    x['scoreidx'] = np.array(np.argsort(np.argsort(x['score'])[::-1]))
    return x


clust_df = clust_df.groupby('filename').apply(f )

clust_df['model'] = None
clust_df['model'][clust_df['filename'].str.contains('ld') & clust_df['filename'].str.contains('xyz') ] = 'srm.xyz'
clust_df['model'][clust_df['filename'].str.contains('.xsoma.')] = 'xsoma'
clust_df['model'][clust_df['filename'].str.contains('.clist.')] = 'clist'
clust_df['model'][clust_df['filename'].str.contains('srm_clist_xsoma')]= 'srm_clist_xsoma'

tgtdf = clust_df #[df['filename'].str.contains('retina.2')]

measure = 'ari'
#ndf = tgtdf[tgtdf['scoreidx']==0]
b = tgtdf.sort(measure, ascending=False).groupby('model', as_index=False).first()

filenames = b['filename']


f = pylab.figure(figsize=(12, 8))

                 
# bins = np.linspace(0, 0.6, 20)
# for fi, filename in enumerate(filenames):
#     ax = f.add_subplot(2, 2, fi+1)
#     subset = tgtdf[tgtdf['filename'] == filename]
#     #print subset.head(1)
#     label = subset.head(1).irow(0)['model']
#     ax.hist(subset[measure], label=label, bins=bins, normed=True)
#     ax.axvline(subset[measure].mean(), c='r', linewidth=4)
#     ax.set_title(label)
#     ax.set_xlim(bins[0], bins[-1])
#     ax.grid()
# f.savefig('pl.comparison.best.ari_coarse.pdf')


# f = pylab.figure(figsize=(12, 8))

                 
# bins = np.linspace(0, 0.6, 40)
# for fi, (gi, subset) in enumerate(tgtdf.groupby('model')):
#     ax = f.add_subplot(2, 2, fi+1)
#     #print subset.head(1)
#     label = gi
#     ax.hist(np.array(subset[measure]), label=label, bins=bins, normed=True)
#     ax.axvline(subset[measure].median(), c='r', linewidth=4)
#     ax.set_title(label)
#     ax.set_xlim(0, 0.5)
#     ax.grid()
# f.savefig('pl.comparison.allparam.ari.pdf')

preddf = pickle.load(open("predlinks.pickle", 'r'))

#print preddf['filename']


preddf['srm'] = preddf['filename'].str.contains('srm')

preddf = preddf[preddf['filename'].str.contains('fixed')]

# get rid of the one-off value

#preddf['model'] = preddf[preddf['filename'].str.contains('simple_bb') == False]
def foo(x):
    s = x['filename'].split('.')
    if len(s) > 5:
        return s[5]
    else:
        return ""

preddf['distvar'] = preddf.apply(foo, axis=1)
print 'here'
preddf['tp'] = preddf['t_t'] / preddf['t_tot']
preddf['fp'] = preddf['f_t'] / preddf['f_tot']
preddf['frac_wrong'] = 1.0 - (preddf['t_t'] + preddf['f_f']) / (preddf['t_tot'] + preddf['f_tot'])
# compute AUC

def g_auc(g):
    g = g.sort('pred_thold', ascending=False)
    # under estimate by using rectangles and anchoring left corner
    widths = np.diff(g['fp'])

    areas = g['tp'][:-1] * widths

    auc = np.sum(areas)

    return pandas.Series({'auc' : auc})


preddf['filename']=preddf['filename'].apply(lambda x : x[:-(len(".predlinks.pickle"))])


aucsdf = preddf.groupby(['filename', 'chain_i']).apply(g_auc).reset_index()
aucsdf['chain_i'] = aucsdf['chain_i'].apply(lambda x : int(x))

aucsdf['distvar'] = aucsdf.apply(foo, axis=1)
aucsdf['srm'] = aucsdf['filename'].str.contains('srm')
aucsdf.set_index(['filename', 'chain_i'], inplace=True)

tgtdf['filename'] = tgtdf['filename'].apply(lambda x : x[:-(len('.samples'))])

tgtdf.set_index(['filename', 'chain_i'], inplace=True)

pickle.dump({'aucsdf' : aucsdf, 
             'tgtdf' : tgtdf}, 
            open("dummy.pickle", 'w'))


#alldf = aucsdf.merge(tgtdf, how='inner')
#print alldf
