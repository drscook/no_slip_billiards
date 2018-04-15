import math
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets

### Global variables
abs_tol = 1e-5
rel_tol = 1e-3
np_dtype = np.float64
threads_per_block_max = 1024
sqrt_threads_per_block_max = int(np.floor(np.sqrt(threads_per_block_max)))


def next_state(part):
    get_col_time(part)
        
    if np.isinf(part.dt):
        raise Exception("No future collisions detected")

    part.t += part.dt
    part.pos += part.vel * part.dt

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

    

### Helper functions

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

        
### Classes

# master wall class; subclass for each wall shape
class Wall():
    def get_pw_gap(self, p=Ellipsis):        
        return self.get_pw_col_coefs(gap_only=True)
    
class FlatWall(Wall):
    def __init__(self, base_point, normal):
        self.dim = dim
        self.base_point = base_point
        self.normal_static = make_unit(normal)
    
    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos - self.base_point
        nu = self.normal_static
        c = dx.dot(nu) - part.radius
        c[np.isinf(c)] = np.inf #corrects -np.inf to +np.inf
        c[np.isnan(c)] = np.inf #corrects np.nan to +np.inf
        if gap_only == True:
            return c
        dv = part.vel
        b = dv.dot(nu)
        a = np.zeros(b.shape, dtype=b.dtype)
        return a, b, c
    
    def resolve_pw_collision(self, part, p):
        nu = self.normal_static
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu



class Particles():
    def __init__(self):
        self.dim = dim
        self.num = num_part
        self.radius = np.full(num_part, radius, dtype=np_dtype)
        self.mass = np.full(num_part, mass, dtype=np_dtype)
        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1)
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.vel = np.full([self.num, self.dim], np.inf, dtype=np_dtype)        

        self.t = 0.0
        self.col = {}
        N = max(self.num, len(wall))
        self.default_mask = np.array([N,N], dtype=np.int32)
        self.pp_mask = self.default_mask.copy()
        self.pw_mask = self.default_mask.copy()
        
        self.t_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.col_hist = []
        
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
            z = rnd.uniform(-side+self.radius[p], side-self.radius[p], size=dim)
            self.pos[p] = z.copy()
            if self.check_pos() == True:
#                 print('Placed particle {}'.format(p))
                return
        raise Exception('Could not place particle {}'.format(p))

    def rand_vel(self, p):
        self.vel[p] = rnd.normal(0.0, 1.0, size=dim)

    def initialize(self):
        for p in range(self.num):
            if np.any(np.isinf(self.pos[p])):
                self.rand_pos(p)
            if np.any(np.isinf(self.vel[p])):
                self.rand_vel(p)    
    
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
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        nu = part.pos[p2] - part.pos[p1]
        nu = make_unit(nu)
        dv = part.vel[p2] - part.vel[p1]
        w = dv.dot(nu) * nu
        part.vel[p1] += 2 * (m2/M) * w
        part.vel[p2] -= 2 * (m1/M) * w

    def get_KE(self):
        KE = self.mass[:,np.newaxis] * (self.vel**2)
        return np.sum(KE) / 2

    def check(self):
        if self.check_pos(soft=True) == False:
            print(self.pw_gap)
            print(self.pp_gap)
            raise Exception('A particle escaped')
        if abs(1-self.KE_init/self.get_KE()) > rel_tol:
            raise Exception(' KE not conserved')
        return True
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos.copy())
        self.vel_hist.append(self.vel.copy())
        self.col_hist.append(self.col)

    def clean_up(self):
        part.t_hist = np.asarray(part.t_hist)
        part.pos_hist = np.asarray(part.pos_hist)
        part.vel_hist = np.asarray(part.vel_hist)

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
