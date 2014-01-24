import numpy as np
from matplotlib import pylab
import model

# GIBBS SAMPLING
def die_roll(v):
    """
    Take in a vector of probs and roll
    """
    x = np.cumsum(v)
    r = np.random.rand()
    return np.searchsorted(x, r)

def scores_to_prob(x):
    """
    Take in a vector of scores
    normalize, log-sumpadd, and return
    
    """
    xn = x - np.max(x)
    a = np.logaddexp.accumulate(xn)[-1]
    xn = xn - a
    return np.exp(xn)

def sample_from_scores(scores):
    return die_roll(scores_to_prob(scores))


def gibbs_sample(domain_inf, rng, impotent=False):

    T_N = domain_inf.entity_count()

    if impotent:
        print "gibbs_sample: IMPOTENT"

    for entity_pos in np.random.permutation(T_N):
        g = domain_inf.remove_entity_from_group(entity_pos)
        if domain_inf.group_size(g) == 0:
            temp_group = g
        else:
            temp_group = domain_inf.create_group(rng)


        groups = domain_inf.get_groups()
        scores = np.zeros(len(groups))
        for gi, group_id in enumerate(groups):
            scores[gi] = domain_inf.post_pred(group_id, entity_pos)
        #print entity_pos, scores
        sample_i = sample_from_scores(scores)
        new_group = groups[sample_i]

        if impotent:
            new_group = g

        domain_inf.add_entity_to_group(new_group, entity_pos)
        if new_group != temp_group:
            assert domain_inf.group_size(temp_group) == 0
            domain_inf.delete_group(temp_group)

def gibbs_sample_nonconj(domain_inf, M, rng, impotent=False):
    """
    Radford neal Algo 8 for non-conj models
    
    M is the number of ephemeral clusters
    
    We assume that every cluster in the model is currently used
    
    impotent: if true, we always assign the object back to its original
    cluster. Useful for benchmarking
    
    """
    T_N = domain_inf.entity_count()

    if impotent:
        print "gibbs_sample_nonconj IMPOTENT"

    if T_N == 1:
        return # nothing to do 

    for entity_pos in range(T_N):
        g = domain_inf.remove_entity_from_group(entity_pos)
        extra_groups = []
        if domain_inf.group_size(g) == 0:
            extra_groups.append(g)
        while len(extra_groups) < M:
            extra_groups.append(domain_inf.create_group(rng))

        groups = domain_inf.get_groups()
        scores = np.zeros(len(groups))
        for gi, group_id in enumerate(groups):
            scores[gi] = domain_inf.post_pred(group_id, entity_pos)
            # correct the score for the empty groups
            if group_id in extra_groups:
                scores[gi] -= np.log(M)

        # DEBUGGING
        # normed_scores = scores_to_prob(scores)
        # # top five 
        # sorted_scores_i = np.argsort(normed_scores)[::-1]
        # pylab.figure(figsize=(6, 12))
        # pylab.subplot( 2, 1,  1)
        # bins = np.linspace(-0.3, 1.3, 100)

        # pylab.hist(domain_inf.features['f1'].data[entity_pos], 
        #            bins)
        # pylab.subplot( 2, 1,  2)

        # sorted_scores_i = sorted_scores_i[:8]
        # for si, s in enumerate(sorted_scores_i):
        #     gid = groups[s]
        #     ss = domain_inf.features['f1'].components[gid]
        #     p = model.compute_mm_probs(bins,  zip(ss['pi'], ss['mu'], ss['var']))
            
        #     pylab.plot(bins[:-1], p, linewidth=3, 
        #                alpha = (float(len(sorted_scores_i)) - si)/len(sorted_scores_i), 
        #                c = 'k')

        # pylab.show()
        #print entity_pos, scores
        sample_i = sample_from_scores(scores)
        if impotent: 
            new_group = g
        else:
            new_group = groups[sample_i]

        domain_inf.add_entity_to_group(new_group, entity_pos)
        for eg in extra_groups:
            if domain_inf.group_size(eg) == 0:
                domain_inf.delete_group(eg)

        # for r in domain_inf.relations:
        #     r.assert_assigned()

