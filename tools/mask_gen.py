import numpy as np    
mask = np.zeros((227,227))
mask[70:120,90:150]=1
import scipy.miscscipy.misc.imsave('mask.jpg', mask)
