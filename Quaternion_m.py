import numpy as np
class Quaternion(object):
    def __init__(self):
        self.w = 1.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.norm = 1.0

def qPrint(q):
    return q.w, q.x, q.y, q.z
    
def qProduct(p, q):
    qnew = Quaternion()
    qnew.w = p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z
    qnew.x = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y
    qnew.y = p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x
    qnew.z = p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w
    return qnew

def qNormalize(q):
    norm = np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
    return norm

def qConjugate(p):
    q = Quaternion()
    q.w =  p.w
    q.x = -p.x
    q.y = -p.y
    q.z = -p.z
    return q

def qInverse(p):
    qInv = Quaternion()
    qInv = qConjugate(p)/qNormalize(p)
    return qInv

def q2Angle(q):
    angle = 2.0*np.arccos(q.w)
    return angle

def q2axis(q):
    angle = q2Angle(q)
    norm = 1.0/np.sin(angle/2.0)
    axis = np.zeros((3))
    axis[0] = q.x*norm
    axis[1] = q.y*norm
    axis[2] = q.z*norm
    return axis

def q2matrix(q):
    matrix = np.zeros((3,3))

    matrix[0,0] = 1.0 - 2.0 * ( q.y*q.y + q.z*q.z )
    matrix[0,1] =       2.0 * ( q.x*q.y - q.w*q.y )
    matrix[0,2] =       2.0 * ( q.x*q.z + q.w*q.z )
    matrix[1,0] =       2.0 * ( q.x*q.y + q.w*q.z )
    matrix[1,1] = 1.0 - 2.0 * ( q.x*q.x + q.z*q.z )
    matrix[1,2] =       2.0 * ( q.y*q.z + q.w*q.x )
    matrix[2,0] =       2.0 * ( q.x*q.z - q.w*q.y )
    matrix[2,1] =       2.0 * ( q.y*q.z + q.w*q.x )
    matrix[2,2] = 1.0 - 2.0 * ( q.x*q.x + q.y*q.y )

    return matrix

def q2DCM(q):
    # https://www.mathworks.com/help/aeroblks/quaternionstodirectioncosinematrix.html
    matrix = np.zeros((3,3))

    matrix[0,0] = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z 
    matrix[0,1] =       2.0 * ( q.x*q.y - q.w*q.z )
    matrix[0,2] =       2.0 * ( q.x*q.z + q.w*q.y )

    matrix[1,0] =       2.0 * ( q.x*q.y + q.w*q.z )
    matrix[1,1] = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z 
    matrix[1,2] =       2.0 * ( q.y*q.z - q.w*q.x )

    matrix[2,0] =       2.0 * ( q.x*q.z - q.w*q.y )
    matrix[2,1] =       2.0 * ( q.y*q.z + q.w*q.x )
    matrix[2,2] = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z 

    return matrix

def q_plus_s(q, s):
    qn = Quaternion()
    qn.w = q.w + s
    qn.x = q.x + s
    qn.y = q.y + s
    qn.z = q.z + s
    return qn

def q_plus_q(p, q):
    qn = Quaternion()
    qn.w = p.w + q.w
    qn.x = p.x + q.x
    qn.y = p.y + q.y
    qn.z = p.z + q.z
    return qn

def q_minus_s(q, r):
    qn = Quaternion()
    qn.w = q.w - r
    qn.x = q.x - r
    qn.y = q.y - r
    qn.z = q.z - r
    return qn

def s_times_q(r, q):
    qn = Quaternion()
    qn.w = q.w*r
    qn.x = q.x*r
    qn.y = q.y*r
    qn.z = q.z*r
    return qn

def q_times_s(q, r):
    qn = Quaternion()
    qn.w = q.w*r
    qn.x = q.x*r
    qn.y = q.y*r
    qn.z = q.z*r
    return qn

def qdiv(q,r):
    qn = Quaternion()
    qn.w = q.w/r
    qn.x = q.x/r
    qn.y = q.y/r
    qn.z = q.z/r
    return qn

def rot2q(angle, u):
    qn = Quaternion()
    qn.w  = np.cos(angle/2.0)
    normu = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    normq = np.sin(angle/2.0)/normu
    qn.x = u[0]*normq
    qn.y = u[1]*normq
    qn.z = u[2]*normq
    return qn

def vec2q(v):
    qn = Quaternion()
    normv = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    a = normv
    if(abs(a) < 1E-9):
            qn.w = 1.0
            qn.x = 0.0
            qn.y = 0.0
            qn.z = 0.0
            return qn
    qn.w  = np.cos(a/2.0)
    normq = np.sin(a/2.0)/a
    qn.x  = v[0]*normq
    qn.y  = v[1]*normq
    qn.z  = v[2]*normq
    return qn

def rotate_q(vec, q):
    temp = Quaternion()
    temp.w = 0.0
    temp.x = vec[0]
    temp.y = vec[1]
    temp.z = vec[2]
    # v' = q v q*
    newq   = qProduct( q, qProduct(temp, qConjugate(q)) )
    # print(qfromv.w)
    # print(vqstar)
    # print(vqstar.w)
    v = np.zeros((3))
    v[0] = newq.x
    v[1] = newq.y
    v[2] = newq.z
    return v

def rotate_au(vVec, angle, uVec):
    v = np.zeros((3))
    v = rotate_q(vVec, rot2q(angle,uVec))
    return v

def rotate_a(vec, da):
    normda = np.sqrt(da[0]**2 + da[1]**2 + da[2]**2)
    v = rotate_q( vec, rot2q( normda, da ) )
    return v

# The following is from wikipediea...
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def euler2quat(Attitude, iEuler):
    q1 = Quaternion()
    q2 = Quaternion()
    q3 = Quaternion()
    q  = Quaternion()

    for i in range(3):
        if iEuler[i] == 1:
            ThetaX = Attitude[i]
        elif iEuler[i] == 2:
            ThetaY = Attitude[i]
        elif iEuler[i] == 3:
            ThetaZ = Attitude[i]
        else:
            import sys
            sys.stop(f'Error!! invalid value of iEuler({i})={iEuler[i]}')

    q1.w = np.cos(ThetaX/2.0)
    q1.x = np.sin(ThetaX/2.0)

    q2.w = np.cos(ThetaY/2.0)
    q2.y = np.sin(ThetaY/2.0)

    q3.w = np.cos(ThetaZ/2.0)
    q3.z = np.sin(ThetaZ/2.0)

    for i in range(2,0,-1):
        if iEuler[i] == 0:
            if(i == 2):
                q = q1
            else:
                q = qProduct(q1, q)
                q = q_times_s(q, 1.0/qNormalize(q))

        if iEuler[i] == 1:
            if (i == 2):
                q = q2
            else:
                q = qProduct(q2, q)
                q = q_times_s(q, 1.0/qNormalize(q))

        if iEuler[i] == 2:
            if (i == 2):
                q = q3
            else:
                q = qProduct(q3, q)
                q = q_times_s(q, 1.0/qNormalize(q))
    return q

def quat2euler(quat):
    # print(f'inside quat2euler. {qPrint(quat)}')
    # X-axis rotation
    t0 = 2.0*(quat.w*quat.x + quat.y*quat.z)
    t1 = 1.0 - 2.0*(quat.x*quat.x + quat.y*quat.y)
    Roll = np.arctan2(t0, t1)

    # Y-axis rotation
    t2 = 2.0*(quat.w*quat.y - quat.z*quat.x)
    if (t2 >= 1.0):
        t2 = +1.0
    Yaw = np.arcsin(t2)

    # Z-axis rotation
    t3 = 2.0*(quat.w*quat.z + quat.x*quat.y)
    t4 = 1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z)
    Pitch = np.arctan2(t3, t4)

    # print(f'Pitch={Pitch}, Yaw={Yaw}, Roll={Roll}')
    attitude=np.zeros((3))
    attitude[0] = Pitch
    attitude[1] = Yaw
    attitude[2] = Roll
    return attitude

def test_quat():
    # vector to rotate
    v = np.array([1.0, 0.0, 0.0])

    # vector to rotate about
    v2 = np.array([0.0, 0.0, 1.0])

    # Angle to rotate
    theta = np.radians(20.0)

    # Create quaternion to rotate
    q = Quaternion()
    q = rot2q(angle=theta, u=v2)

    np.set_printoptions(precision=3)
    print(f'q2angle gives the angle (in radians): {q2Angle(q=q)*180.0/np.pi}')
    q = rot2q(angle=theta, u=v2)
    print(f'rot2q creates unit quaternion: {q.w, q.x, q.y, q.z}')
    print(f'-------------------------')
    print(f'Original vector v: {v}')
    print(f'Rotating about v2: {v2}')
    print(f'By an angle of: {theta} radians or {np.degrees(theta)} degrees')
    print(f'rotate_q rotates vector v using unit quaternion q:     {rotate_q(vec=v, q=q)}')
    print(f'rotate_q(rot2q) rotates v using angle and real vector: {rotate_q(vec=v, q=rot2q(theta,v2))}')
    print(f'vec2q(v)                                             : {qPrint(vec2q(v))}')
    print(f'rot2q(theta,v2))                                     : {qPrint(rot2q(theta,v2))}')
    print(f'qprod(vec2q(v),rot2q(theta,v2))                      : {qPrint(qProduct( vec2q(v), rot2q(theta,v2) ))}')
    print(f'q2axis(qprod(vec2q(v),rot2q(theta,v2)))              : {q2axis( qProduct( vec2q(v), rot2q(theta,v2) ) )}')
    print(f'q2axis(qprod(rot2q(theta,v2),vec2q(v)))              : {q2axis( qProduct( rot2q(theta,v2),  vec2q(v)) )}')
