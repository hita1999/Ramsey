import numpy as np
import numpy.linalg as LA

#red edges = 0 
#blue edges = 1
a = np.array([[0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 1],
              [1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [1, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0],])

b = np.array([[0,1,1],
              [1,0,1],
              [1,1,0]])

w,v = LA.eig(a)
wb, vb = LA.eig(b)

w_sorted = np.sort(w)[::-1]
wb_sorted = np.sort(wb)[::-1]

#print(w_sorted)
print(np.round(w_sorted, 10))

print(np.round(wb_sorted, 10))
#print(v)