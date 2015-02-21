import numpy as np
from matplotlib import pylab
import irm
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle    
import sys
sys.path.append("../code")
import linkplots
import util

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))

np.random.seed(0)

SIDE_N = 6

# class_conn = {(0, 1) : ('d', 1.0, 0.7), 
#               (1, 2) : ('d', 2.0, 0.8), 
#               (3, 2) : ('p', 0.1), 
#               (3, 0) : ('d', 1.7, 0.9), 
#               (1, 3) : ('p', 0.1)}


# nodes_with_class, connectivity = irm.data.generate.c_mixed_dist_block(SIDE_N, 
#                                                                       class_conn, JITTER=0.1, rand_conn_prob=0.01)


# class_conn = {(0, 1) : (1.0, 0.8, 0.1), 
#               (1, 2) : (2.0, 0.4, 0.3), 
#               (3, 2) : (0.7, 0.9, 0.2), 
#               (3, 0) : (1.7, 0.7, 0.5), 
#               (1, 3) : (0.5, 0.5, 0.2)}


# nodes_with_class, connectivity = irm.data.generate.c_bump_dist_block(SIDE_N, 
#                                                                       class_conn, JITTER=0.3, rand_conn_prob=0.01)



class_conn = {(0, 1) : (0.7, 0.8), 
              (1, 2) : (2.2, 0.6), 
              #(3, 2) : (0.7, 0.9), 
              (2, 1) : (0.2, 0.8), 
              (0, 3) : (1.4, 0.7), 
              (3, 4) : (2.4, 0.8), 
              (4, 0) : (1.0, 0.3), 
              }


nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, 
                                                                     class_conn, JITTER=0.3, default_param=0.01)



CELL_N = len(connectivity)

conn_and_dist = np.zeros((CELL_N, CELL_N), 
                         dtype = [('link', np.uint8), 
                                  ('distance', np.float32)])

for ni, (ci, posi) in enumerate(nodes_with_class):
    for nj, (cj, posj) in enumerate(nodes_with_class):
        conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
        conn_and_dist[ni, nj]['distance'] = dist(posi, posj)



f = pylab.figure()
ax = f.add_subplot(1, 1, 1)
ai = np.random.permutation(CELL_N)
c_r = connectivity[ai, :]
c_r = c_r[:, ai]

ax.imshow(c_r, interpolation='nearest', cmap=pylab.cm.Greys)

ax.set_xticks([])
ax.set_yticks([])
f.savefig("source.f1.raw.pdf")

cd = conn_and_dist['distance'][ai, :]
cd = cd[:, ai]
ax.imshow(cd, interpolation='nearest', cmap=pylab.cm.Greys)

ax.set_xticks([])
ax.set_yticks([])
f.savefig("source.f1.distance.pdf")


c_class = nodes_with_class['class']

# now create synthetic distances for each class
class_syn_dist = { 0 : (0.2, 0.5, 0.7), 
                   1 : (0.3,), 
                   2 : (0.55, 0.6), 
                   3 : (0.8,), 
                   4 : (0.1, 0.9)}
class_soma_pos_var = { 0 : (0.1, 0.05), 
                       1 : (0.3, 0.01), 
                       2 : (0.5, 0.03), 
                       3 : (0.85, 0.1), 
                       4 : (0.3, 0.04)
}

N_SYN = 40
syn_pro_bins = np.linspace(0, 1.0, 30)
syn_pro_hist = np.zeros((len(c_class), len(syn_pro_bins)-1))
soma_depth = np.zeros(len(c_class))

for ci, c in enumerate(c_class):
    pos = []
    for i in range(N_SYN):
        pi = np.random.randint(0, len(class_syn_dist[c]))
        p = np.random.normal(class_syn_dist[c][pi], 0.07)
        pos.append(p)
    b, _= np.histogram(pos, bins=syn_pro_bins)
    syn_pro_hist[ci] = b
    mu, std = class_soma_pos_var[c]
    soma_depth[ci] = np.random.normal(mu, std)

f = pylab.figure()
ax = f.add_subplot(1, 1, 1)
ax.imshow(syn_pro_hist, interpolation='nearest', cmap=pylab.cm.Greys)

ax.set_xticks([])
ax.set_yticks([])
for i in np.argwhere(np.diff(c_class) != 0).flatten():
    ax.axhline(i + 0.5)
ax.set_aspect(0.5)
f.savefig("source.f1.synprof.pdf")

f = pylab.figure()
ax = f.add_subplot(1, 1, 1)

ax.imshow(syn_pro_hist[ai], interpolation='nearest', cmap=pylab.cm.Greys)

ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect(0.5)
f.savefig("source.f1.synprof.raw.pdf")


f = pylab.figure(figsize=(2, 6))
ax = f.add_subplot(1, 1, 1)
ax.scatter(soma_depth, 
           np.arange(len(c_class)), edgecolor='none', c='k')
ax.set_xlim(0, 1)
ax.set_ylim(len(c_class), 0)
ax.set_xticks([])
ax.set_yticks([])
for i in np.argwhere(np.diff(c_class) != 0).flatten():
    ax.axhline(i + 0.5)
f.savefig("source.f1.somadepth.pdf")


f = pylab.figure(figsize=(2, 6))
ax = f.add_subplot(1, 1, 1)
ax.scatter(soma_depth[ai], 
           np.arange(len(c_class)), edgecolor='none', c='k')
ax.set_xlim(0, 1)
ax.set_ylim(len(c_class), 0)
ax.set_xticks([])
ax.set_yticks([])
f.savefig("source.f1.somadepth.raw.pdf")



CLASS_N = len(np.unique(c_class))

from brewer2mpl import qualitative
bmap = qualitative.Set1[CLASS_N]
print bmap
group_colors = bmap.mpl_colors # (np.linspace(0, 1, CLASS_N))

ai = np.argsort(c_class).flatten()
c_sorted = connectivity[ai, :]
c_sorted = c_sorted[:, ai]

f2 = pylab.figure()
ax = f2.add_subplot(1, 1, 1)

c_class_sorted = c_class[ai]
di = np.argwhere(np.diff(c_class_sorted) > 0).flatten()
ax.imshow(c_sorted, interpolation='nearest', cmap=pylab.cm.Greys)
for d in di:
    ax.axhline(d+0.5)
    ax.axvline(d+0.5)
ax.set_xticks([])
ax.set_yticks([])
f2.savefig("source.f1.sorted.pdf")

f2 = pylab.figure()

ax = f2.add_subplot(1, 1, 1)
mat = c_sorted
ax.imshow(mat, cmap=pylab.cm.Blues, interpolation='nearest')
borders =  di # np.argwhere(np.diff(ca[cell_order])).flatten()

ax.set_xticks([])
ax.set_yticks([])

#group_colors = np.random.permutation(group_colors)
prev = 0

BORDER_WIDTH = 2
N = mat.shape[0]
for b, fc in zip(borders.tolist()  + [mat.shape[0]], group_colors):
    
    ax.plot([prev, b], [-BORDER_WIDTH, -BORDER_WIDTH], c=fc, linewidth=10, solid_capstyle="butt")
    
    ax.plot([-BORDER_WIDTH, -BORDER_WIDTH], [prev, b], c=fc, linewidth=10, solid_capstyle="butt")
    ax.plot([prev, b], [N+BORDER_WIDTH, N+BORDER_WIDTH], c=fc, linewidth=10, solid_capstyle="butt")
    ax.plot([N+BORDER_WIDTH, N+BORDER_WIDTH], [prev, b], c=fc, linewidth=10, solid_capstyle="butt")    
    prev = b
    
for p in borders:
    ax.axhline(p + 0.5, c='k', linewidth=1)
    ax.axvline(p + 0.5, c='k', linewidth=1)
#ax.set_ylim(BORDER_WIDTH + N, )
f2.savefig("source.f1.sortedcolors.pdf")

# hilariously construct suffstats and hps by hand
hps = {'mu_hp' : 1.0, 
       'lambda_hp' : 1.0, 
       'p_max' : 0.9, 
       'p_min' : 0.0001}
ss = {}
for c1 in range(CLASS_N):
    for c2 in range(CLASS_N):
        c = (c1, c2)
        if c in class_conn:
            mu = class_conn[c][0]
            lamb = class_conn[c][0]/12
        else:
            mu = 0.0001
            lamb = 0.0001

        ss[c] = {'mu' : mu, 'lambda' : lamb}


f3 = pylab.figure()
irm.plot.plot_t1t1_params(f3, conn_and_dist, nodes_with_class['class'], 
                          ss, hps, model="LogisticDistance", MAX_DIST=3.5)
f3.savefig("source.f1.latent.pdf")



dists = np.array([0.5, 1.0, 2.0])

mats = np.zeros((len(dists), CLASS_N, CLASS_N))


for di, d in enumerate(dists):
    e = irm.irmio.latent_distance_eval(d, ss, hps, 'LogisticDistance')

    # convert to canonical 
    for (g1, g2), p in e.iteritems():
        c1 = g1
        c2 = g2 
        mats[di, c1, c2] = p



#group_colors_opaque = group_colors.copy()
#group_colors_opaque[:, -1] = 1.0
#group_colors = ['r', 'g', 'b', 'y', 'k']


fig = pylab.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, aspect='equal') 


def color_func(x):
    if x < 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0, x**3])

def width_func(x):
    return 7.0
linkplots.plot_helper(ax, mats, group_colors, color_func, link_width_func = width_func, 
                      X_SCALE=3.0, plot_colorbar=False, x_pad =2, directed=True)
ax.set_xticklabels([""] + ["%2.1f um"% (x*40) for x in  dists], fontsize=20)


ax.text(6.0*len(mats)+1, 2, "conn\nprob", fontsize= 14 )
ax.text(6.0*len(mats)+2.2, 2, "0.0", fontsize= 14 )

ax.text(6.0*len(mats)+2.2, 4, "1.0", fontsize= 14 )


fig.savefig("source.f1.connections.pdf")


# X_SCALE = 4.0
# CIRCLE_SIZE = 0.4
# for i in range(len(mats)):
#     points = []
#     colors = []
#     widths = []
    
#     patches = []
#     m = mats[i]
#     for ci in range(m.shape[0]):
#         for cj in range(m.shape[1]):
#             if ci <= cj   and ci != cj:
#                 p = m[ci, cj]
#                 if p >= 0.1:
#                     points.append([[i*X_SCALE, ci], [(i+1)*X_SCALE, cj]])
#                     colors.append([0, 0, 0, p**3])
#                     #print p
#                     widths.append(4.0)
#         circle = Circle((i*X_SCALE,ci), CIRCLE_SIZE)
#         patches.append(circle)

#     lc = LineCollection(points, colors = colors, linewidths=np.array(widths), zorder=1)
#     ax.add_collection(lc)

#     p = PatchCollection(patches, color = group_colors, edgecolor='k', zorder=10) # , color='w', edgecolor='k')

#     ax.add_collection(p)
# # last ones
# patches = [Circle((mats.shape[0]*X_SCALE,ci), CIRCLE_SIZE) for ci in range(mats.shape[1])]
# p = PatchCollection(patches, color = group_colors)
# ax.add_collection(p)

# ax.set_yticks([])
# ax.set_xticks(np.arange(mats.shape[0] + 1) * X_SCALE)
# ax.set_xticklabels([""] + ["%2.1f um"% x for x in  dists], fontsize=20)
# ax.set_xlim(-1, len(mats)*X_SCALE + 1)
# ax.set_ylim(mats[0].shape[0], -1)
# ax.set_xlabel("cell distance", fontsize=20)
# ax.set_ylabel("discovered cell type", fontsize=20)

# fig.savefig("source.f1.connections.pdf")

# DIST = 0.5

# circos_p = irm.plots.circos.CircosPlot(c_class, "0.5r", "70p", 
#                                            ['red', 'green', 'blue', 'yellow', 'purple'])

# v = irm.irmio.latent_distance_eval(DIST, 
#                                    ss, hps, 'LogisticDistance')

# thold = 0.3
# ribbons = []
# links = []
# pairs_plotted = set()

# ribbons = []
# for (src, dest) in v.keys():
#     p1 = v[(src, dest)]
#     p2 = v[(dest, src)]
#     p = max(p1, p2)
#     if (src, dest) in pairs_plotted or (dest, src) in pairs_plotted:
#         pass
#     else:
#         if p > thold :
#             pix = int(10*p)
#             print src, dest, p, pix

#             ribbons.append((src, dest, pix))

#     pairs_plotted.add((src, dest))

# circos_p.add_class_ribbons(ribbons)
# circos_p.add_plot('scatter', {'r0' : '1.0r', 
#                               'r1' : '1.15r', 
#                               'min' : 0, 
#                               'max' : 1.1, 
#                               'glyph' : 'circle', 
#                               'glyph_size' : 12, 
#                               'color' : 'black',
#                               'stroke_thickness' : 0
#                           }, 
#                   soma_depth, 
#                   {'backgrounds' : [('background', {'color': 'vvlgrey', 
#                                                     'y0' : 0.0, 
#                                                     'y1' : 1.0})],  
#                    'axes': [('axis', {'color' : 'vlgrey', 
#                                       'thickness' : 1, 
#                                       'spacing' : '%fr' % 0.1})]})


# for bi, b in enumerate(syn_pro_bins[:-1]):
#     width = 0.15/20.
#     start = 1.15 + width*bi
#     end = start + width
#     circos_p.add_plot('heatmap', {'r0' : '%fr' % start, 
#                                   'r1' : '%fr' % end, 
#                                   'stroke_thickness' : 0, 
#                                   'color' : 'greys-6-seq'}, 
#                       syn_pro_hist[:, bi])

# irm.plots.circos.write(circos_p, "source.f1.circos.png")


# DISTS = [0.1, 1.0, 2.5]
# for dist_i, dist_threshold in enumerate(DISTS):
#     circos_p = irm.plots.circos.CircosPlot(c_class, "0.5r", "70p", 
#                                            ['red', 'green', 'blue', 'yellow', 'purple'])

#     v = irm.irmio.latent_distance_eval(dist_threshold, 
#                                        ss, hps, 'LogisticDistance')

#     thold = 0.3
#     ribbons = []
#     links = []
#     pairs_plotted = set()

#     ribbons = []
#     for (src, dest) in v.keys():
#         p1 = v[(src, dest)]
#         p2 = v[(dest, src)]
#         p = max(p1, p2)
#         if (src, dest) in pairs_plotted or (dest, src) in pairs_plotted:
#             pass
#         else:
#             if p > thold :
#                 pix = int(10*p)
#                 print src, dest, p, pix

#                 ribbons.append((src, dest, pix))

#         pairs_plotted.add((src, dest))

#     circos_p.add_class_ribbons(ribbons)

#     irm.plots.circos.write(circos_p, "source.f1.circos.%d.svg" % dist_i)




