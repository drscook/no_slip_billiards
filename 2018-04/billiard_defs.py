### Core dynamical code for seriel mode

import math
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from timeit import default_timer as timer

### Global variables
abs_tol = 1e-5
rel_tol = 1e-3
BOLTZ = 1.0
np_dtype = np.float64
threads_per_block_max = 1024
sqrt_threads_per_block_max = int(np.floor(np.sqrt(threads_per_block_max)))

def get_col_time(part):
    part.get_pw_col_time_cpu()
    if mode == 'parallel':
        get_pp_col_time_gpu(part)
    else:
        part.get_pp_col_time_cpu()
    part.dt = min([np.min(part.pp_dt), np.min(part.pw_dt)])
    return part.dt


def next_state(part):
    get_col_time(part)
        
    if np.isinf(part.dt):
        raise Exception("No future collisions detected")

    part.t += part.dt
    part.pos += part.vel * part.dt
    part.pos_loc += part.vel * part.dt

    pw_events = (part.pw_dt - part.dt) < 1e-7
    pw_counts = contract(pw_events)
    pw_tot = np.sum(pw_counts)
    
    pp_events = (part.pp_dt - part.dt) < 1e-7
    pp_counts = contract(pp_events)
    pp_tot = np.sum(pp_counts)
    
    if (pw_tot == 0) & (pp_tot == 2):
        p, q = np.nonzero(pp_counts)[0]
        part.col = {'p':p, 'q':q}
        part.pw_mask[:] = part.default_mask[:]
        part.pp_mask[:] = [p,q]
        part.resolve_pp_collision(p, q)
    elif (pw_tot == 1) & (pp_tot == 0):
        p = np.argmax(pw_counts)
        w = np.argmax(pw_events[p])
        part.col = {'p':p, 'w':w}
        part.pw_mask[:] = [p,w]
        part.pp_mask[:] = part.default_mask[:]
        wall[w].resolve_pw_collision(part, p)
    else:
        P = tuple(np.nonzero(pp_counts + pw_counts)[0])
        print('COMPLEX COLLISION DETECTED. Re-randomizing positions of particles {}'.format(P))
        part.pos[P,:] = np.inf
        part.pw_mask[:] = part.default_mask[:]
        part.pp_mask[:] = part.default_mask[:]
        for p in P:
            part.rand_pos(p)

def initialize(wall, part):
    if np.all([w.dim == part.dim for w in wall]) == False:
        raise Exception('Not all wall and part dimensions agree')
    if np.all((part.gamma >= 0) & (part.gamma <= np.sqrt(2/part.dim))) == False:
        raise Exception('illegal mass distribution parameter {}'.format(gamma))

    for (i, w) in enumerate(wall):
        w.idx = i
        w.pw_gap_min = w.gap_m * part.radius + w.gap_b        

    part.mom_inert = part.mass * (part.gamma * part.radius)**2
    part.sigma_lin = np.sqrt(BOLTZ * part.temp / part.mass)
    part.sigma_spin = np.sqrt(BOLTZ * part.temp / part.mom_inert)
    part.pp_gap_min = cross_subtract(part.radius, -part.radius)
    np.fill_diagonal(part.pp_gap_min, -1)


    for p in range(part.num):
        if np.any(np.isinf(part.pos[p])):
            part.rand_pos(p)
        if np.any(np.isinf(part.vel[p])):
            part.rand_vel(p)
        if np.any(np.isinf(part.orient[p])):
            part.orient[p] = np.eye(part.dim)
        if np.any(np.isinf(part.spin[p])):
            part.rand_spin(p)

    part.pos_loc = part.pos.copy()
            
    part.KE_init = part.get_KE()
    part.check()
    part.record_state()



        
### Classes


class PW_CollisionLaw:
    @staticmethod
    def resolve_collision(self, wall, part, p):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

class PW_SpecularLaw(PW_CollisionLaw):
    name = 'PW_SpecularLaw'
    def resolve_collision(self, wall, part, p):
        nu = wall.normal(part.pos_loc[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu
        
class PW_IgnoreLaw(PW_CollisionLaw):
    name = 'PW_IgnoreLaw'
    def resolve_collision(self, wall, part, p):
        pass

class PW_TerminateLaw(PW_CollisionLaw):
    name = 'PW_TerminateLaw'
    def resolve_collision(self, wall, part, p):
        raise Exception('particle {} hit termination wall {}'.format(p, wall.idx))

class PW_PeriodicLaw(PW_CollisionLaw):
    name = 'PW_PeriodicLaw'
    def __init__(self, wrap_dim, wrap_wall):
        self.wrap_dim = wrap_dim
        self.wrap_wall = wrap_wall

    def resolve_collision(self, wall, part, p):
        d = self.wrap_dim   # which dim will have sign flip
        part.pos_loc[p, d] *= -1   # flips sign of dim d
        part.pw_mask[:] = [p, self.wrap_wall]

#No-slip law in any dimension from private correspondence with Cox and Feres.
#See last pages of: https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/feres_N_dim_no_slip_law_2017.pdf
# Uses functions like Lambda_nu defined at the end of this file
class PW_NoSlipLaw(PW_CollisionLaw):
    name = 'PW_NoSlipLaw'
    def resolve_collision(self, wall, part, p):
        nu = wall.normal(part.pos_loc[p])
        m = part.mass[p]
        g = part.gamma[p]
        r = part.radius[p]
        d = (2*m*g**2)/(1+g**2)
        
        U_in = part.spin[p]
        v_in = part.vel[p]
        U_out = U_in - (d/(m*g**2) * Lambda_nu(U_in, nu)) + (d/(m*r*g**2)) * E_nu(v_in, nu)
        v_out = (r*d/m) * Gamma_nu(U_in, nu) + v_in - 2 * Pi_nu(v_in, nu) - (d/m) * Pi(v_in,nu)

        part.spin[p] = U_out
        part.vel[p] = v_out

class PP_CollisionLaw:
    @staticmethod
    def resolve_collision(self, part, p1, p2):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

class PP_IgnoreLaw(PP_CollisionLaw):
    name = 'PP_IgnoreLaw'
    def resolve_collision(self, part, p1, p2):
        pass

class PP_SpecularLaw(PP_CollisionLaw):
    name = 'PP_SpecularLaw'        
    def resolve_collision(self, part, p1, p2):
        nu = part.pos[p2] - part.pos[p1]
        nu = make_unit(nu)
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2

        dv = part.vel[p2] - part.vel[p1]
        w = dv.dot(nu) * nu
        part.vel[p1] += 2 * (m2/M) * w
        part.vel[p2] -= 2 * (m1/M) * w

class PP_NoSlipLaw(PP_CollisionLaw):
    name = 'PP_NoSlipLaw'
    def resolve_collision(self, part, p1, p2):
        nu = part.pos[p2] - part.pos[p1]
        nu = make_unit(nu)
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        
        r1 = part.radius[p1]
        r2 = part.radius[p2]        
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        
        U1_in = part.spin[p1]
        U2_in = part.spin[p2]
        v1_in = part.vel[p1]
        v2_in = part.vel[p2]

        U1_out = (U1_in-d/(m1*g1**2) * Lambda_nu(U1_in, nu)) \
                    + (-d/(m1*r1*g1**2)) * E_nu(v1_in, nu) \
                    + (-r2/r1)*(d/(m1*g1**2)) * Lambda_nu(U2_in, nu) \
                    + d/(m1*r1*g1**2) * E_nu(v2_in, nu)

        v1_out = (-r1*d/m1) * Gamma_nu(U1_in, nu) \
                    + (v1_in - 2*m2/M * Pi_nu(v1_in, nu) - (d/m1) * Pi(v1_in, nu)) \
                    + (-r2*d/m1) * Gamma_nu(U2_in, nu) \
                    + (2*m2/M) * Pi_nu(v2_in, nu) + (d/m1) * Pi(v2_in, nu)

        U2_out = (-r1/r2)*(d/(m2*g2**2)) * Lambda_nu(U1_in, nu) \
                    + (-d/(m2*r2*g2**2)) * E_nu(v1_in, nu) \
                    + (U2_in - (d/(m2*g2**2)) * Lambda_nu(U2_in, nu)) \
                    + (d/(m2*r2*g2**2)) * E_nu(v2_in, nu)

        v2_out = (r1*d/m2) * Gamma_nu(U1_in, nu) \
                    + (2*m1/M) * Pi_nu(v1_in, nu) + (d/m2) * Pi(v1_in, nu) \
                    + (r2*d/m2) * Gamma_nu(U2_in, nu) \
                    + v2_in - (2*m1/M) * Pi_nu(v2_in, nu) - (d/m2) * Pi(v2_in,nu)
        part.spin[p1] = U1_out
        part.spin[p2] = U2_out
        part.vel[p1] = v1_out
        part.vel[p2] = v2_out   


# master wall class; subclass for each wall shape
class Wall():
    def __init__(self):
        self.dim = dim
        self.gap_b = 0.0
        self.gap_m = 1.0
        self.temp = 1.0
        self.collision_law = PW_SpecularLaw()
    
    def get_pw_gap(self, p=Ellipsis):        
        return self.get_pw_col_coefs(gap_only=True)
    
    def resolve_pw_collision(self, part, p):
        self.collision_law.resolve_collision(self, part, p)

    @staticmethod
    def normal(pos):
        raise Exception('You should implement the method normal() in a subclass.')

    @staticmethod
    def get_pw_col_coefs(self, part):
        raise Exception('You should implement the method get_pw_col_time() in a subclass.')

    @staticmethod
    def get_mesh():
        raise Exception('You should implement the method get_mesh() in a subclass.')

class FlatWall(Wall):
    def __init__(self, base_point, normal, tangents):
        super().__init__()
        self.name = 'flat'        
        self.base_point = np.asarray(base_point, dtype=np_dtype)
        self.normal_static = make_unit(normal)
        self.tangents = np.asarray(tangents, dtype=np_dtype)
        
    def normal(self, pos):
        return self.normal_static
    
    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos_loc - self.base_point
        nu = self.normal_static
        c = dx.dot(nu) - self.pw_gap_min
        c[np.isinf(c)] = np.inf #corrects -np.inf to +np.inf
        c[np.isnan(c)] = np.inf #corrects np.nan to +np.inf
        if gap_only == True:
            return c
        dv = part.vel
        b = dv.dot(nu)
        a = np.zeros(b.shape, dtype=b.dtype)
        return a, b, c
    
    def get_mesh(self):
        self.mesh = flat_mesh(self.tangents) + self.base_point  # see visualize.py

class SphereWall(Wall):
    def __init__(self, base_point, radius):
        super().__init__()
        self.name = 'sphere'        
        self.base_point = np.asarray(base_point, dtype=np_dtype)
        self.radius = radius
        self.gap_b = radius

    def normal(self, pos): # normal depends on collision point
        dx = pos - self.base_point
        return make_unit(dx)  # see below for make_unit

    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos_loc - self.base_point
        c = contract(dx**2)
        if gap_only == True:
            return np.sqrt(c) - self.pw_gap_min
        c -= self.pw_gap_min**2
        dv = part.vel
        b = contract(dx*dv) * 2
        a = contract(dv**2)
        return a, b, c

    def get_mesh(self):
        self.mesh = sphere_mesh(self.dim, self.radius) + self.base_point # see visualize.py

class Particles():
    def __init__(self):
        self.dim = dim
        self.num = num_part
        self.mass = np.full(self.num, 1.0, dtype=np_dtype)
        self.radius = np.full(self.num, 1.0, dtype=np_dtype)
        gamma = {'uniform':np.sqrt(2/(2+self.dim)), 'shell':np.sqrt(2/self.dim), 'point':0.0}
        self.gamma = np.full(self.num, gamma['uniform'], dtype=np_dtype)
        self.temp = np.full(self.num, 1.0, dtype=np_dtype)
        
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.pos_loc = self.pos.copy()
        self.vel = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        
        self.dim_spin = int(self.dim * (self.dim - 1) / 2)
        self.orient = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)
        self.spin = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)

        self.t = 0.0
        self.col = {}
        N = max(self.num, len(wall))
        self.default_mask = np.array([N,N], dtype=np.int32)
        self.pp_mask = self.default_mask.copy()
        self.pw_mask = self.default_mask.copy()
        
        self.collision_law = PP_SpecularLaw()
        
        self.t_hist = []
        self.pos_hist = []
        self.vel_hist = []
#         self.orient_hist = []
        self.spin_hist = []
        self.col_hist = []

    def get_mesh(self):
        S = sphere_mesh(self.dim, 1.0)
        if self.dim == 2:
            S = np.vstack([S, [-1,0]])

        part.mesh = []
        for p in range(part.num):
            self.mesh.append(S*self.radius[p]) # see visualize.py
        part.mesh = np.asarray(part.mesh)
            
        # pretty color for visualization
        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , self.num).round().astype(int)
        self.clr = [cm(i) for i in idx]
        
    def get_pp_col_coefs(self, gap_only=False):
        dx = cross_subtract(part.pos)
        c = np.einsum('pqd, pqd -> pq', dx, dx)
        if gap_only == True:
            return np.sqrt(c) - self.pp_gap_min
        c -= self.pp_gap_min**2
        dv = cross_subtract(self.vel)
        b = 2*np.einsum('pqd, pqd -> pq', dv, dx)
        a =   np.einsum('pqd, pqd -> pq', dv, dv)
        return a, b, c

    def get_pp_gap(self):
        self.pp_gap = self.get_pp_col_coefs(gap_only=True)
        return self.pp_gap

    def get_pw_gap(self):
        self.pw_gap = np.array([w.get_pw_col_coefs(gap_only=True) for w in wall]).T
        return self.pw_gap

    def check_pos(self, soft=False):
        if soft == True:
            tol = -abs_tol
        else:
            tol = abs_tol
        self.get_pp_gap()
        self.get_pw_gap()
        return np.all(self.pp_gap > tol) and np.all(self.pw_gap > tol)

    def rand_pos(self, p):
        max_attempts = 1000
        for k in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(-cell_size[d], cell_size[d])
            self.pos_loc[p] = self.pos[p].copy()
            if self.check_pos() == True:
#                 print('Placed particle {}'.format(p))
                return
        raise Exception('Could not place particle {}'.format(p))

    def rand_vel(self, p):
        self.vel[p] = rnd.normal(0.0, self.sigma_lin[p], size=dim)
    
    def rand_spin(self, p):
        v = [rnd.normal(0.0, self.sigma_spin[p]) for d in range(self.dim_spin)]
        self.spin[p] = spin_mat_from_vec(v)
        
    def get_pp_col_time_cpu(self):
        a, b, c = self.get_pp_col_coefs()
        self.pp_dt_full = solve_quadratic(a, b, c, mask=swap(self.pp_mask))
        self.pp_dt = np.min(self.pp_dt_full, axis=1)
        return a, b, c
        
    def get_pw_col_time_cpu(self):
        a = np.zeros([self.num, len(wall)])
        b = a.copy()
        c = a.copy()
        for (j,w) in enumerate(wall):
            a[:,j], b[:,j], c[:,j] = w.get_pw_col_coefs()
        self.pw_dt = solve_quadratic(a, b, c, mask=tuple(self.pw_mask))
        return a, b, c
    
    def resolve_pp_collision(self, p1, p2):
        self.collision_law.resolve_collision(self, p1, p2)

    def get_KE(self):
        self.KE_lin = self.mass * contract(self.vel**2)
        self.KE_ang = self.mom_inert * contract(self.spin**2) / 2
        self.KE = (self.KE_lin + self.KE_ang) / 2
        return np.sum(self.KE)

    def check_angular(self):
#         o_det = np.abs(np.linalg.det(self.orient))-1
#         orient_check = np.abs(o_det) < abs_tol
        skew = self.spin + np.swapaxes(self.spin, -2, -1)
        spin_check = contract(skew*skew) < abs_tol
        return np.all(spin_check) #and np.all(orient_check)
    
    def check(self):
        if self.check_pos(soft=True) == False:
            print(self.pw_gap)
            print(self.pp_gap)
            raise Exception('A particle escaped')
        if abs(1-self.KE_init/self.get_KE()) > rel_tol:
            raise Exception(' KE not conserved')
        if part.check_angular() == False:
            raise Exception('A particle has invalid orintation or spin matrix')
        return True
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos.copy())
        self.vel_hist.append(self.vel.copy())
#         self.orient_hist.append(self.orient.copy())
        self.spin_hist.append(self.spin.copy())
        self.col_hist.append(self.col)

    def clean_up(self):
        part.t_hist = np.asarray(part.t_hist)
        part.pos_hist = np.asarray(part.pos_hist)
        part.vel_hist = np.asarray(part.vel_hist)
#         part.orient_hist = np.asarray(part.orient_hist)
        part.spin_hist = np.asarray(part.spin_hist)
        part.num_steps = len(part.t_hist)
        part.num_frames = part.num_steps

def solve_quadratic(a, b, c, mask=[]):
    # The hard task is finding the first root.  Because the sum and product of the two roots must equal
    # -b/a and c/a resp, we can compute the second easily.
    # We use a combination of the Citardauq and the quadratic formulas.  Each is numerically stable for a
    # different range of coeficients.
    small = np.full(a.shape, np.inf)
    big = small.copy()
    M = 1e8
    d = b**2 - 4*a*c  #discriminant
    real = (d >= 0)
    d[real] = np.sqrt(d[real]) * np.sign(b)[real]
    
    e = -(b + d) / 2
    citardauq_formula = real & (np.abs(c) < M * np.abs(e))
    small[citardauq_formula] = c[citardauq_formula] / e[citardauq_formula]
    
    f = -(b - d) / 2
    quadratic_formula = real & ~citardauq_formula & (np.abs(f) < M * np.abs(a))
    small[quadratic_formula] = f[quadratic_formula] / a[quadratic_formula]
    
    s = real & (np.abs(b) < M * np.abs(a))
    big[s] = -b[s]/a[s] - small[s]
    
    with np.errstate(invalid='ignore'):  # suppresses warnings for inf-inf
        g = a * small
        m = real & ~s & (np.abs(c) < M * np.abs(g))
    big[m] = c[m] / g[m]

    try:
        small[mask] = big[mask]
        big[mask] = np.inf
    except:
        pass
    
    small_idx = small < 0
    big_idx = big < 0
    clear_idx = small_idx & big_idx
    small[clear_idx] = np.inf    
    
    swap_idx = small_idx & ~big_idx
    small[swap_idx] = big[swap_idx]
    
#     big[big_idx] = np.inf
    return small#, big



### Helper functions



#######################################################################################################
### No-Slip Collision Functions ###
#######################################################################################################
def spin_mat_from_vec(v):
    # Converts spin vector to spin matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Exponential_map
                     
    l = len(v)
    # l = d(d-1) -> d**2 - d - 2l = 0
    d = (1 + np.sqrt(1 + 8*l)) / 2
    if d % 1 != 0:
        raise Exception('vector {} of length {} converts to dim = {:.2f}.  Not integer.'.format(v,l,d))
    d = int(d)
    M = np.zeros([d,d])
    idx = np.triu_indices_from(M,1)
    s = (-1)**(np.arange(len(v))+1)
    w = v * s
    w = w[::-1]
    M[idx] = w
    M = make_symmetric(M, skew=True)
    return M

def spin_vec_from_mat(M):
    idx = np.triu_indices_from(M,1)
    w = M[idx]
    s = (-1)**(np.arange(len(w))+1)
    w = w[::-1]    
    v = w * s
    return v
   
def wedge(a,b):
    return np.outer(b,a)-np.outer(a,b)

def Pi_nu(v, nu):
    return v.dot(nu) * nu

def Pi(v, nu):
    w = Pi_nu(v ,nu)
    return v - w

def Lambda_nu(U, nu):
    return wedge(nu, U.dot(nu))

def E_nu(v, nu):
    return wedge(nu, v)

def Gamma_nu(U, nu):
    return U.dot(nu)
    
#######################################################################################################
###  Random Functions ###
#######################################################################################################

def random_uniform_sphere(num=1, dim=2, radius=1.0):
    pos = rnd.normal(size=[num, dim])
    pos = make_unit(pos, axis=1)
    return abs(radius) * pos


def random_uniform_ball(num=1, dim=2, radius=1.0):
    pos = random_uniform_sphere(num, dim, radius)
    r = rnd.uniform(size=[num, 1])
    return r**(1/dim) * pos
    

def swap(L):
    a, b = L
    return ((a,b),(b,a))

def cross_subtract(u,v=None):
    if v is None:
        v=u.copy()
    with np.errstate(invalid='ignore'):  # suppresses warnings for inf-inf
        w = u[:,np.newaxis] - v[np.newaxis,:]
        w[np.isnan(w)] = np.inf
    return w

def contract(A, keepdims=[0]):
    # sum all dimensions except those in keepdims
    keepdims = listify(keepdims)
    A = np.asarray(A)
    return np.einsum(A, range(A.ndim), keepdims)
           
def make_unit(A, axis=-1):
    # Normalizes along given axis so the sum of squares is 1.
    A = np.asarray(A, dtype=np_dtype)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M

def make_symmetric(A, skew=False):
    """
    Returns symmetric or skew-symmatric matrix by copying upper triangular onto lower.
    """
    A = np.asarray(A)
    U = np.triu(A,1)
    if skew == True:
        return U - U.T
    else:
        return np.triu(A,0) + U.T    

def listify(X):
    # Ensure X is a list
    if isinstance(X, list):
        return X
    elif (X is None) or (X is np.nan):
        return []
    elif isinstance(X,str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]




## The code below was an experiment to find "optimal" choices for quadratic.  Couldn't make it work, though
# def solve_quadratic_new(a, b, c, mask=[]):
#     # The hard task is finding the first root.  Because the sum and product of the two roots must equal
#     # -b/a and c/a resp, we can compute the second easily.
#     # We use a combination of the Citardauq and the quadratic formulas.  Each is numerically stable for a
#     # different range of coeficients.
#     small = np.full(a.shape, np.inf)
#     big = small.copy()
#     M = 1e8
#     d = b**2 - 4*a*c  #discriminant
#     real = (d >= 0)
#     d[real] = np.sqrt(d[real]) * np.sign(b)[real]
    
#     with np.errstate(invalid='ignore'):
#         cit = -(b + d) / 2    
#         quad = -(b - d) / 2

#         cit_margin = mag_dif(c, cit)
#         quad_margin = mag_dif(quad, a)

#         cit_idx  = real & (cit_margin > quad_margin)
#         quad_idx = real & (cit_margin < quad_margin)
#         first = cit_idx | quad_idx

#         small[cit_idx]  = c[cit_idx] / cit[cit_idx]
#         small[quad_idx] = quad[quad_idx] / a[quad_idx]

#         sum_margin = mag_dif(b, a)
#         g = a * small
#         prod_margin = mag_dif(c, g)

#         sum_idx  = first & (sum_margin > prod_margin)
#         prod_idx = first & (sum_margin < prod_margin)

#         big[sum_idx]  = -b[sum_idx] / a[sum_idx] - small[sum_idx]
#         big[prod_idx] = c[prod_idx] / g[prod_idx]

#     try:
#         small[mask] = big[mask]
#         big[mask] = np.inf
#     except:
#         pass
    
#     small_idx = small < 0
#     big_idx = big < 0
#     clear_idx = small_idx & big_idx
#     small[clear_idx] = np.inf    
    
#     swap_idx = small_idx & ~big_idx
#     small[swap_idx] = big[swap_idx]
    
# #     big[big_idx] = np.inf
#     return small#, big



# def mag_dif(a, b):
#     x = np.abs(a)
#     y = np.abs(b)
#     d = x.copy()
#     idx = y > x
#     d[~idx] = -np.inf
#     d[idx] = y[idx] - x[idx]
#     return d
