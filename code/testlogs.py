import numpy as np
from matplotlib import pylab
import time

import irm

N = 1e8
d = np.arange(1, N, dtype=np.float32)/N

lc = irm.pyirm.LogCompare(d.tolist())


t1 = time.time()
lc.log()
t2 = time.time()
print "Log took", t2-t1, "secs"

t1 = time.time()
lc.logf()
t2 = time.time()
print "Logf took", t2-t1, "secs"


t1 = time.time()
lc.fastlog()
t2 = time.time()
print "fastlog took", t2-t1, "secs"

t1 = time.time()
lc.fasterlog()
t2 = time.time()
print "fasterlog took", t2-t1, "secs"
