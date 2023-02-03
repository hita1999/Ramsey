import numpy as np
import numpy.linalg as LA

a = np.array([[0, 0, 1, 0, 1, 1],
              [0, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [1, 0, 0, 1, 0, 0],
              [1, 1, 0, 1, 0, 0],])

w,v = LA.eig(a)

w_sorted = np.sort(w)[::-1]

print(np.round(w_sorted, 10))
#print(v)