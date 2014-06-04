import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from ruffus import * 
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
vblpcm = importr("VBLPCM")
network = importr("network")


import numpy as np
from matplotlib import pylab
import cPickle as pickle
import pandas

import irm
import others


@files(None, 'test.data.pickle')
def create_data(_, outfile):

    SIDE_N = 5
    GROUP_N = 4
    conn = {}
    for i in range(GROUP_N):
        conn[(i, i)] = (10.0, 0.1)
        conn[(i, (i + 1) % GROUP_N)] = (1.0, 0.2)
        conn[(i, (i + 2) % GROUP_N)] = (1.5, 0.7)


    nodes, obs = irm.data.generate.c_class_neighbors(SIDE_N , conn)
    obs = obs.astype(float)
    dist_mat = np.zeros(obs.shape, dtype=np.float)
    # compute the distance matrix
    for n1_i, n1 in enumerate(nodes):
        for n2_i, n2 in enumerate(nodes):
            dist_mat[n1_i, n2_i] = np.sum((nodes[n1_i]['pos'] - nodes[n2_i]['pos'])**2)
        
    
    
    pickle.dump({'nodes' : nodes, 
                 'obs' : obs, 
                 'dist_mat' : dist_mat}, 
                open(outfile, 'w'))
    
@files(create_data, 'test.data.png')
def plot_data(infile, outfile):

    d = pickle.load(open(infile, 'r'))
    obs = d['obs']

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(obs, interpolation='nearest', cmap=pylab.cm.Greys)
    f.savefig(outfile)

@files(create_data, 'test.output.pickle')
def run(infile, outfile):
    d = pickle.load(open(infile, 'r'))
    obs = d['obs']
    dist_mat = d['dist_mat']
    results = []
    # now try and generate the data from four gaussians
    for g in range(4, 10):
        for d in range (2, 5):
            for dm_i, dist_mat in enumerate([None, 1./dist_mat, dist_mat]):
                #perm = np.random.permutation(len(obs))
                assignments = others.vblpcm_cluster(obs, g, d, dist_mat)
                results.append({'d' : d, 'g' : g, 
                                'assignments' : assignments, 
                                'dm_i' : dm_i})
    outdf = pandas.DataFrame(results)

    pickle.dump(outdf, open(outfile, 'w'))
                           
@files([create_data, run], 'results.png')
def agg_results((in_data, out_run), plot_filename):
    data = pickle.load(open(in_data, 'r'))
    obs = data['obs']
    outdf = pickle.load(open(out_run, 'r'))
    
    true_assign = data['nodes']['class']
    for row_i, row in outdf.iterrows():
        assignments = np.array(row['assignments'])
        ai = np.argsort(assignments)

        f = pylab.figure()
        ax = f.add_subplot(1, 2, 1)
        ax.imshow(obs, interpolation='nearest', cmap=pylab.cm.Greys)
        ais = assignments[ai]
        for r in np.argwhere(np.diff(true_assign) != 0).flatten() :
            ax.axhline(r + 0.5, c='b')
            ax.axvline(r + 0.5, c='b')

        ax = f.add_subplot(1, 2, 2)
        obs_sorted = obs[ai]
        obs_sorted = obs_sorted[:, ai]
        ax.imshow(obs_sorted, interpolation='nearest', cmap=pylab.cm.Greys)
        ais = assignments[ai]
        for r in np.argwhere(np.diff(ais) != 0).flatten() :
            ax.axhline(r + 0.5, c='b')
            ax.axvline(r + 0.5, c='b')

        f.savefig('results.%s.png' % row_i, dpi=300)

if __name__ == "__main__":
    pipeline_run([run, plot_data, agg_results])
