import numpy as np
import cPickle as pickle
import sys

for fname in sys.argv[1:]:
    d = pickle.load(open(fname))
    goodchains = len(d['chains'])
    if "1000" in fname and goodchains < 15:
        print fname
