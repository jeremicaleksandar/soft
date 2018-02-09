#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
import numpy as np

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def da_li_se_seku(a1,a2, b1, b2) :
	if (a1[0] == a2[0] and a1[1] == a2[1]):
		return 0

	presek = seg_intersect(a1, a2, b1, b2)
	leva = max(min(a1[0], a2[0]), min(b1[0], b2[0]))
	desna = min(max(a1[0], a2[0]), max(b1[0], b2[0]))
	donja = max(min(a1[1], a2[1]), min(b1[1], b2[1]))
	gornja = min(max(a1[1], a2[1]), max(b1[1], b2[1]))
	if leva <= presek[0] <= desna and donja <= presek[1] <= gornja:
	    return 1
	return 0

	
# p1 = np.array( [0.0, 0.0] )
# p2 = np.array( [1.0, 0.0] )

# p3 = np.array( [4.0, -5.0] )
# p4 = np.array( [4.0, 2.0] )

# print (da_li_se_seku( p1,p2, p3,p4))

# p1 = np.array( [2.0, 2.0] )
# p2 = np.array( [4.0, 3.0] )

# p3 = np.array( [6.0, 0.0] )
# p4 = np.array( [6.0, 3.0] )

# print (da_li_se_seku( p1,p2, p3,p4))

# p1 = np.array( [0.0, 0.0] )
# p2 = np.array( [3.0, 3.0] )

# p3 = np.array( [0.0, 4.0] )
# p4 = np.array( [4.0, 0.0] )

# print (da_li_se_seku( p1,p2, p3,p4))