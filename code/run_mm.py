import numpy as np
from matplotlib import pylab
import model
import mixmodel
import irm
import gibbs

# range is always [0, 1]
# synth_comps = [[(1.0, 0.1, 0.1)], 
#                [(1.0, 0.9, 0.1)]]

np.random.seed(0)


synth_comps = [[(0.5, 0.1, 0.05),
                (0.5, 0.9, 0.05)], 
               [(0.2, 0.4, 0.05), 
                (0.8, 0.6, 0.05)]]


GROUP_N = len(synth_comps)
BIN_N = 10
BINS = np.linspace(0, 1.0, BIN_N + 1)
BIN_WIDTH = BINS[1] - BINS[0]

# now generate the fake data 
DP_N = 100
ENTITIES_PER_GROUP = 50
ROW_N = ENTITIES_PER_GROUP * GROUP_N
data = np.zeros((ROW_N, BIN_N), dtype=np.int32)

# now generate the fake data:
for ci, comp in enumerate(synth_comps):
    p = model.compute_mm_probs(BINS, comp)
    p = p / np.sum(p)

    data[ci*ENTITIES_PER_GROUP:(ci+1)*ENTITIES_PER_GROUP] = np.random.multinomial(DP_N, p, size=ENTITIES_PER_GROUP)

    
# now let's do some fucking inference
data = np.random.permutation(data)



MODEL = model.BinnedDist(BIN_N, 2, 0.05)
f = mixmodel.Feature(data, MODEL)

mm = mixmodel.MixtureModel(ROW_N, {'f1' : f})

rng = None
# random init
grp = {}
for i, g in enumerate(np.random.permutation(np.arange(ROW_N) % 10)):
    if g not in grp:
        grp[g] = mm.create_group(rng)
    mm.add_entity_to_group(grp[g], i)
    
print mm.score()

for i in range(100):
    gibbs.gibbs_sample_nonconj(mm, 20, rng)
    for group_id, comp in f.components.iteritems():
        di = list(f.assignments[group_id])
        
        new_ss = model.mh_comp(MODEL, f.hps, comp, f.data[di])
        f.components[group_id] = new_ss
    
    print i, mm.score(), irm.util.count(mm.get_assignments()).values()
    

a = mm.get_assignments()
ai = np.argsort(a).flatten()
fig = pylab.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(data, interpolation='nearest')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(data[ai], interpolation='nearest')
for i in np.argwhere(np.diff(a[ai]) > 0).flatten():
    ax.axhline(i+0.5)

f2 = pylab.figure(figsize=(4, 8))
for ci, ss in enumerate(f.components.values()):
    ax = f2.add_subplot(len(f.components), 1, ci+1)
    plot_bins = np.linspace(0, 1.0, 500)
    ss_z = zip(ss['pi'], ss['mu'], ss['var'])
    print ss_z
    p = model.compute_mm_probs(plot_bins, ss_z)
    ax.plot(plot_bins[:-1], p)
        
pylab.show()
