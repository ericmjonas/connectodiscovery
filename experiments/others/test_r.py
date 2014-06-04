import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

a = robjects.r.list(10)
robjects.r.assign("foo", a)
b = robjects.r.list(10)
robjects.r.assign("foo2", b)
robjects.r.save("foo", "foo2", file="test.rda")
