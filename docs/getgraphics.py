import numpy as np
import re
import sys
fname = sys.argv[1]

texfile = open(fname, 'r').read()

r = re.compile("\\includegraphics(\[.+\])*\{([\w0-9\./\-]+)\}")
g = [a[1] for a in r.findall(texfile)]
for f in g:
    print f, 
