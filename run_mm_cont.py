import numpy as np
from matplotlib import pylab
import model
import mixmodel
import irm
import gibbs
import cPickle as pickle
from sklearn import metrics

from ruffus import * 

# range is always [0, 1]
# synth_comps = [[(1.0, 0.1, 0.01)], 
#                [(1.0, 0.9, 0.01)]]

np.random.seed(0)


def create_data():
    synth_comps = [[(0.5, 0.1, 0.01),
                    (0.5, 0.9, 0.1)], 
                   [(0.2, 0.4, 0.01), 
                    (0.8, 0.6, 0.01)], 
                   [(0.8, 0.4, 0.01), 
                    (0.2, 0.6, 0.01)], 
                   [(0.6, 0.4, 0.01), 
                    (0.2, 0.6, 0.01),
                    (0.2, 0.9, 0.001)],
                   [(0.25, 0.2, 0.001), 
                    (0.25, 0.4, 0.001),
                    (0.25, 0.6, 0.001),
                    (0.25, 0.8, 0.001)], 
               ]

    def gen_synth_comps():
        N = 20
        out = []
        for i in range(N):
            outparam = []
            K = np.random.poisson(3) + 1
            weights = np.random.dirichlet(np.ones(K)*0.5)
            for k in range(K):
                outparam.append((weights[k], 
                                 np.random.rand(), 
                                 np.random.uniform(0.001, 0.05)))
            out.append(outparam)
        return out

    synth_comps = gen_synth_comps()
    print synth_comps

    GROUP_N = len(synth_comps)

    # now generate the fake data 
    ENTITIES_PER_GROUP = 20
    ROW_N = ENTITIES_PER_GROUP * GROUP_N
    data = []
    true_source = []
    BIN_N = 100
    BINS = np.linspace(0, 1.0, BIN_N + 1)
    # now generate the fake data:
    for ci, comp in enumerate(synth_comps):
        for ei in range(ENTITIES_PER_GROUP):
            dp_n = np.random.poisson(170)

            data.append(model.sample_from_mm(dp_n, comp))
            true_source.append(ci)

    hist_view = np.zeros((ROW_N, BIN_N))
    for row_i, row in enumerate(data):

        x, _ = np.histogram(row, bins=BINS)
        hist_view[row_i] = x
    # pylab.imshow(hist_view, interpolation='nearest')
    # pylab.show()

    return data, true_source

def load_data():
    featuredf  = pickle.load(open('features.pickle'))['featuredf']
    X_MIN = 60
    X_MAX = 130
    data = featuredf['contact_x_list'].tolist()
    normed_data = []
    truth = []
    for d, t in zip(data, featuredf['type_coarse'].tolist()):
        x = (np.array(d) - X_MIN)/(X_MAX-X_MIN)
        if type(x) != np.ndarray:
            x = np.array([x])
        assert len(x) > 0 
        x_finite = x[np.isfinite(x)]
        if len(x_finite) == 0:
            continue

        x[np.isfinite(x) == 0] = np.mean(x_finite)

        assert np.isfinite(x).all()
        normed_data.append(x)
        truth.append(t)

    BIN_N = 100
    BINS = np.linspace(0, 1.0, BIN_N + 1)
    # now generate the fake data:

    hist_view = np.zeros((len(normed_data), BIN_N))
    for row_i, row in enumerate(normed_data):

        x, _ = np.histogram(row, bins=BINS)
        hist_view[row_i] = x
    hist_view = hist_view[np.argsort(truth).flatten()]
    # pylab.imshow(hist_view, interpolation='nearest')
    # pylab.show()

    # transform true source into ints if it is coarse
    ts_ints = []
    ts_pos = {}
    for t in truth:
        if t not in  ts_pos:
            ts_pos[t] = len(ts_pos)
        ts_ints.append(ts_pos[t])
    truth = ts_ints

    return normed_data, truth

@files(None, "mm_cont_data.pickle")
def generate_data(infile, outfile):
    data, true_source = load_data()
    pickle.dump({'data' : data, 
                 'truth' : true_source}, 
                open(outfile, 'w'))


@files(generate_data, 'results.pickle')
def run_exp(infile, outfile):
    d = pickle.load(open(infile, 'r'))

    data, true_source = d['data'], d['truth']


    ROW_N = len(data)

    # now let's do some fucking inference
    order_permutation = np.random.permutation(len(data))

    data = [data[i] for i in order_permutation]
    true_source = np.array(true_source)
    true_source = true_source[order_permutation]


    MODEL = model.MMDist()
    f = mixmodel.Feature(data, MODEL)
    f.hps['comp_k'] = 4
    f.hps['dir_alpha'] = 1.0
    f.hps['var_scale'] = 0.1

    mm = mixmodel.MixtureModel(ROW_N, {'f1' : f})


    INIT_GROUPS = 80
    rng = None
    # random init
    grp = {}
    for i, g in enumerate(np.random.permutation(np.arange(ROW_N) % INIT_GROUPS)):
        if g not in grp:
            grp[g] = mm.create_group(rng)
        mm.add_entity_to_group(grp[g], i)

    print mm.score()
    
    scores = []
    assignments = []

    for i in range(10):
        gibbs.gibbs_sample_nonconj(mm, 20, rng)
        for group_id, comp in f.components.iteritems():
            #di = list(f.assignments[group_id])
            ds = [data[j] for j in f.assignments[group_id]]

            new_ss = model.mh_comp(MODEL, f.hps, comp,  ds)
            f.components[group_id] = new_ss


        scores.append(mm.score())
        assignments.append(mm.get_assignments())
        print i, mm.score(), irm.util.count(mm.get_assignments()).values()

    pickle.dump({'order_permutation' : order_permutation, 
                 'scores' : scores, 
                 'assignments' : assignments, 
                 'data_file' : infile}, 
                open(outfile, 'w'))


@files(run_exp, ['clusters.pdf', 'scores.pdf'])
def plot_results(infile, (clusters_plot, scores_plot)):

    r = pickle.load(open(infile, 'r'))
    data_file = r['data_file']
    assignments = r['assignments']
    scores = r['scores']

    d = pickle.load(open(data_file, 'r'))
    data = d['data']
    
    

    BIN_N = 100
    BINS = np.linspace(0, 1.0, BIN_N + 1)
    ROW_N = len(data)

    hist_view = np.zeros((ROW_N, BIN_N))
    for row_i, row in enumerate(data):

        x, _ = np.histogram(row, bins=BINS)
        hist_view[row_i] = x
    # sort hist by original permutation
    hist_view = hist_view[r['order_permutation']]
    a = assignments[-1]

    ai = np.argsort(a).flatten()
    f = pylab.figure(figsize=(4, 12))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(hist_view, interpolation='nearest')
    ax = f.add_subplot(1, 2, 2)
    ax.imshow(hist_view[ai], interpolation='nearest')
    for i in np.argwhere(np.diff(a[ai]) > 0).flatten():
        ax.axhline(i+0.5, c='w')

    f.savefig(clusters_plot)
    
    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)

    ax.plot(scores)
    f.savefig(scores_plot)

    # f2 = pylab.figure(figsize=(4, 8))

    # for ci, (group_id, ss) in enumerate(f.components.iteritems()):
    #     ax = f2.add_subplot(len(f.components), 1, ci+1)
    #     plot_bins = np.linspace(0, 1.0, 500)
    #     plot_bins_width = plot_bins[1] - plot_bins[0]
    #     ss_z = zip(ss['pi'], ss['mu'], ss['var'])
    #     p = model.compute_mm_probs(plot_bins, ss_z)
    #     p = p / np.sum(p)

    #     ax.plot(plot_bins[:-1], p)
    #     all_group_points = []
    #     for di in np.argwhere(a == group_id):
    #         all_group_points += data[di].tolist()
    #     hist_bins = np.linspace(0, 1.0, 40)
    #     hist_bins_width = hist_bins[1] - hist_bins[0]

    #     h, _ = np.histogram(all_group_points, hist_bins)
    #     h = h.astype(float) / np.sum(h) *(plot_bins_width/hist_bins_width)
    #     pylab.scatter(hist_bins[:-1], h)

    #     # now get the histogram 


if __name__ == "__main__":
    pipeline_run([generate_data, run_exp, 
                  plot_results])
