import math
import itertools as it
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets

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
        w.get_mesh()


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

    part.pos_loc = part.pos.copy()
            
    part.KE_init = part.get_KE()
    part.check()
    part.record_state()


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

def flat_mesh(base_point, tangents):
    pts = 100
    N, D = tangents.shape
    grid = [np.linspace(-1, 1, pts) for n in range(N)]
    grid = np.meshgrid(*grid)
    grid = np.asarray(grid)
    mesh = grid.T.dot(tangents) + base_point    
    return mesh

def sphere_mesh(base_point, radius, dim):
    pts = 100
    grid = [np.linspace(0, np.pi, pts) for d in range(dim-1)]
    grid[-1] *= 2
    grid = np.meshgrid(*grid)                           
    mesh = []
    for d in range(dim):
        w = radius * np.ones_like(grid[0])
        for j in range(d):
            w *= np.sin(grid[j])
        if d < dim-1:
            w *= np.cos(grid[d])
        mesh.append(w)
    return np.asarray(mesh).T + base_point
        
### Classes


class PW_CollisionLaw:
    @staticmethod
    def resolve_collision(self, wall, part, p):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

class PW_SpecularLaw(PW_CollisionLaw):
    name = 'PW_SpecularLaw'
    def resolve_collision(self, wall, part, p):
        nu = wall.normal(part.pos[p])
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
#         s = np.sign(part.pos_loc[p, d]).astype(int)  # is it at + or -        
        part.pos_loc[p, d] *= -1   # flips sign of dim d
        part.pw_mask[:] = [p, self.wrap_wall]
#         part.cell_offset[p, d] += s


# master wall class; subclass for each wall shape
class Wall():
    def __init__(self):
        self.dim = dim
        self.gap_b = 0.0
        self.gap_m = 1.0
        self.temp = 1.0
        self.pw_collision_law = PW_SpecularLaw()
    
    def get_pw_gap(self, p=Ellipsis):        
        return self.get_pw_col_coefs(gap_only=True)
    
    def resolve_pw_collision(self, part, p):
        self.pw_collision_law.resolve_collision(self, part, p)

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
        self.type = 'flat'        
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
        self.mesh = flat_mesh(self.base_point, self.tangents)

class SphereWall(Wall):
    def __init__(self, base_point, radius):
        super().__init__()
        self.type = 'sphere'        
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
        self.mesh = sphere_mesh(self.base_point, self.radius, self.dim)

class Particles():
    def __init__(self):
        self.dim = dim
        self.num = num_part
        self.mass = np.full(self.num, 1.0, dtype=np_dtype)
        self.radius = np.full(self.num, 1.0, dtype=np_dtype)
        self.gamma = np.full(self.num, np.sqrt(2/(2+self.dim)), dtype=np_dtype)
        self.temp = np.full(self.num, 1.0, dtype=np_dtype)
        
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.pos_loc = self.pos.copy()
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
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(-cell_size[d], cell_size[d])
            self.pos_loc[p] = self.pos[p].copy()
            if self.check_pos() == True:
#                 print('Placed particle {}'.format(p))
                return
        raise Exception('Could not place particle {}'.format(p))

    def rand_vel(self, p):
        self.vel[p] = rnd.normal(0.0, 1.0, size=dim)
    
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
