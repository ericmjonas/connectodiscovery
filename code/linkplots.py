import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle
import util


# group_colors_opaque = group_colors.copy()
# group_colors_opaque[:, -1] = 1.0
# fig = pylab.figure(figsize=(20, 6))
# ax = fig.add_subplot(1, 1, 1, aspect='equal') 
# X_SCALE = 6.0
# CIRCLE_SIZE = 0.4

# def color_func(x):
#     return np.array([0.0, 0.0, 0.0, x**3])
# YOFFSET = mats.shape[1]
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
#                 if p >= 0.5:
#                     points.append([[i*X_SCALE, YOFFSET - ci], [(i+1)*X_SCALE, YOFFSET - cj]])
#                     colors.append(color_func(p))
#                     #print p
#                     widths.append((1+p))

    
#     lc = LineCollection(points, colors=colors, linewidths=np.array(widths), zorder=1)
#     ax.add_collection(lc)

# # draw circles
# for i in range(len(mats) + 1):     
#     patches = [Circle((i*X_SCALE,YOFFSET - ci), CIRCLE_SIZE) for ci in range(mats.shape[1])]
#     p = PatchCollection(patches, color = group_colors_opaque, edgecolor='k')
#     ax.add_collection(p)

# ax.set_yticks([])
# ax.set_xticks(np.arange(mats.shape[0] + 1) * X_SCALE)
# ax.set_xticklabels([""] + ["%2.0f um"% x for x in  dists], fontsize=20)
# ax.set_xlim(-1, len(mats)*X_SCALE + 4)
# ax.set_ylim(0, YOFFSET + 1)
# ax.set_xlabel("cell distance", fontsize=20)
# ax.set_ylabel("discovered cell type", fontsize=20)

# color_eval_points = np.linspace(0, 1, 120)
# util.draw_manual_color_bar(ax, color_eval_points, color_func, len(mats)*X_SCALE + 1, 4, 1, 4)


# fig.savefig("tgt.bestsample.connections.pdf")

def draw_links(ax, mats, color_func, X_SCALE=6.0, 
               width_func = None, directed=True, selflinks=False):
    """
    """
    if width_func is None:
        width_func = lambda x: x + 1.0

    YOFFSET = mats.shape[1]
    for i in range(len(mats)):
        points = []
        colors = []
        widths = []

        patches = []
        m = mats[i]
        for ci in range(m.shape[0]):
            for cj in range(m.shape[1]):
                if not directed and ci >= cj:
                    continue
                if not selflinks and ci == cj:
                    continue

                p = m[ci, cj]
                points.append([[i*X_SCALE, YOFFSET - ci],
                               [(i+1)*X_SCALE, YOFFSET - cj]])
                colors.append(color_func(p))
                widths.append(width_func(p))
        lc = LineCollection(points, colors=colors, linewidths=np.array(widths), zorder=1)
        ax.add_collection(lc)

def draw_circles(ax, mats, color_list, X_SCALE=6.0,  CIRCLE_SIZE=0.4):
    YOFFSET = mats.shape[1]

    # draw circles
    for i in range(len(mats) + 1):     
        patches = [Circle((i*X_SCALE,YOFFSET - ci), CIRCLE_SIZE) for ci in range(mats.shape[1])]
        p = PatchCollection(patches, color = color_list, edgecolor='k')
        ax.add_collection(p)

def plot_helper(ax, mats, node_colors, link_color_func, plot_colorbar = True, 
                X_SCALE=6.0, link_width_func=None, x_pad=4, selflinks=False, 
                directed=False): 

    YOFFSET = mats.shape[1]

    draw_links(ax, mats, link_color_func, X_SCALE=X_SCALE, 
                width_func=link_width_func, selflinks=selflinks, directed=directed)
    draw_circles(ax, mats, node_colors, X_SCALE=X_SCALE)
    ax.set_yticks([])
    ax.set_xticks(np.arange(mats.shape[0] + 1) * X_SCALE)
    #ax.set_xticklabels([""] + ["%2.0f um"% x for x in  dists], fontsize=20)
    ax.set_xlim(-1, len(mats)*X_SCALE + x_pad)
    ax.set_ylim(0, YOFFSET + 1)
    ax.set_xlabel("cell distance", fontsize=20)
    ax.set_ylabel("discovered cell type", fontsize=20)

    color_eval_points = np.linspace(np.min(mats), np.max(mats), 120)
    if plot_colorbar:
        util.draw_manual_color_bar(ax, color_eval_points, link_color_func, 
                                   len(mats)*X_SCALE + 1, 4, 1, 4)
    
