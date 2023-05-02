import numpy as np

import warnings
warnings.filterwarnings('error')

try:
    np.mean(np.array([]))
except Warning:
    import pdb; pdb.set_trace()