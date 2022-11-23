import numpy as np
from Quaternion_m import Quaternion, s_times_q, quat2euler, q_plus_q
class rigid_body_dynamics(object):
    def __init__(self):
        import numpy as np
        self.att = [0.0, 0.0, 0.0]
        self.omega_lab = [0.0, 0.0, 0.0]
        self.omega_body = [0.0, 0.0, 0.0]
        self.quaternion = Quaternion()
        self.moments_body = [0.0, 0.0, 0.0]
        self.Imat = [[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]
        self.invImat = np.linalg.inv(self.Imat)
    
    def calculate_alpha(self,w,M,Imat):
        # Solve for angular acceleration
        I_omega = np.matmul(Imat, w)
        # Solve for corss product term due
        omega_cross_I_omega = np.cross( w, I_omega)
        # Solve for angular acceleration
        alpha = np.matmul(np.linalg.inv(Imat), ( M - omega_cross_I_omega))
        return alpha

    def quaternion_rate(self,w,q):
        # Matrix from https://arxiv.org/pdf/0811.2889.pdf
        # matrix = [[-self.x, -self.y, -self.z],
        #           [ self.w, -self.z,  self.y],
        #           [ self.z,  self.w, -self.x],
        #           [-self.y,  self.x,  self.w]]
        # self.quaterion = np.matmul(self.omega_body,0.5*matrix)
        q_dot = Quaternion()
        q_dot.w = (-w[0]*q.x - w[1]*q.y - w[2]*q.z) / 2.0
        q_dot.x = ( w[0]*q.w - w[1]*q.z + w[2]*q.y) / 2.0
        q_dot.y = ( w[0]*q.z + w[1]*q.w - w[2]*q.x) / 2.0
        q_dot.z = (-w[0]*q.y + w[1]*q.x + w[2]*q.w) / 2.0
        return q_dot
        
    def advance_dynamics(self):
        q_k = self.quaternion
        w_k = self.omega_body
        alpha = self.calculate_alpha()

    def compute_rotational_dynamics(self):
        Moments = self.moments_body
        Imat = self.Imat
        q_k0 = self.quaterion
        w_k0 = self.omega_body
        # 1st Step
        q_k = q_k0
        w_k = w_k0
        w_k1 = self.calculate_alpha(w=w_k, M=Moments, Imat=Imat)
        q_k1 = self.quaternion_rate(w=w_k, q=q_k)
        # 2nd Step
        const = 0.5*self.dt
        w_k = w_k0 + const*w_k1
        q_k = q_plus_q(q_k0, s_times_q(const,q_k1))
        w_k2 = self.calculate_alpha(w=w_k, M=Moments, Imat=Imat)
        q_k2 = self.quaternion_rate(w=w_k, q=q_k)
        # 3rd Step
        const = 0.5*self.dt
        w_k = w_k0 + const*w_k2
        q_k = q_plus_q(q_k0, s_times_q(const,q_k2))
        w_k3 = self.calculate_alpha(w=w_k, M=Moments, Imat=Imat)
        q_k3 = self.quaternion_rate(w=w_k, q=q_k)
        # 4th Step
        const = 1.0*self.dt
        w_k = w_k0 + const*w_k3
        q_k = q_plus_q(q_k0, s_times_q(const,q_k3))
        w_k4 = self.calculate_alpha(w=w_k, M=Moments, Imat=Imat)
        q_k4 = self.quaternion_rate(w=w_k, q=q_k)
        # Update values
        temp = q_plus_q( q_k1, s_times_q(2.0,q_k2) )
        temp = q_plus_q( temp, s_times_q(2.0,q_k3))
        temp = q_plus_q( temp, q_k4)
        temp = s_times_q(self.dt/6.0, temp)
        self.quaterion  = q_plus_q(q_k0, temp)
        self.omega_body = w_k0 + self.dt*(w_k1 + 2.0*w_k2 + 2.0*w_k3 + w_k4) / 6.0
        q = self.quaterion
        datt = quat2euler(q)
        self.att[0] = datt[0]
        self.att[1] = datt[1]
        self.att[2] = datt[2]
