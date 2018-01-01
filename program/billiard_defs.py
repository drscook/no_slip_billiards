from helper import *

BOLTZ_CONST = 5

# Helper functions for no slip collision
def Pi_nu(v, nu):
    """
    Projection of v onto n (where n MUST be unit length).
    """
    return v.dot(nu) * nu

def Pi(v, nu):
    """
    Component of v orthogonal to n (where n MUST be unit length).
    """
    w = Pi_nu(v ,nu)
    return v - w


def wedge(a,b):
    """
    Wedge product
    """
    return np.outer(b,a)-np.outer(a,b)

def Lambda_nu(U, nu):
    return wedge(nu, U.dot(nu))

def E_nu(v, nu):
    return wedge(nu, v)

def Gamma_nu(U, nu):
    return U.dot(nu)

def cross_subtract(u, v=None):
    """
    u and v must have same shape except in first slot.  Say u.shape = [m,...] and v.shape = [m,...].  Return w has shape [m,n,...] where w[i,j,...] = u[i,...] - v[j,...].
    """
    if v is None:
        v=u.copy()
    w = u[np.newaxis,:] - v[:,np.newaxis]
    return w

class WallClass():
    def pw_specular_law(self, part, p):
        """
        Particle-wall specular law in any dimension.
        """
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * Pi_nu(part.vel[p], nu)
    
    def pw_no_slip_law(self, part, p):  
        """
        Particle-wall no-slip law in any dimension, per Renato Feres.
        """
        nu = self.normal(part.pos[p])
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

    def resolve_collision(self, part, p):
        if self.collision_law == 'specular':
            self.pw_specular_law(part, p)
        elif self.collision_law == 'no_slip':
            self.pw_no_slip_law(part, p)
        elif self.collision_law == 'thermal':
            raise Exception('Thermal law not yet implemented')
        else:
            raise Exception('Unknown pw collision law {}'.format(self.collision_law))
    
class FlatWall(WallClass):
    def __init__(self, pos, half_length, half_width, normal=None, basis=None, name='flat', collision_law='specular'):
        self.name = name
        self.pos = np.asarray(pos).astype(float)
        self.half_length = half_length
        self.half_width = half_width
        if normal is not None:
            self.normal_static = munit(normal).astype(float)  # defined in helper

        if basis is not None:
            self.basis = munit(basis, ax=0)
            self.normal_static = self.basis[:,-1]
        else:
            self.basis = make_orth_norm_basis(self.normal_static)  # defined in helper
            self.basis[[0,-1]] = self.basis[[-1,0]]
            self.basis = self.basis.T  # basis vectors now in written down columns, normal rightmost
            
        if(len(self.pos) == len(self.normal_static)):
            self.dim = len(self.pos)
        else:
            raise Exception("Shape mismatch: position {} and normal {}".format(pos.shape, normal.shape))

        self.collision_law = collision_law
        
    def parametrize(self):
        grid_u = np.linspace(-1, 1, 10)
        grid_v = np.linspace(-1, 1, 10)
        grid_u, grid_v = np.meshgrid(grid_u, grid_v)
        vect_u = self.basis[:,0] * self.half_length
        vect_v = self.basis[:,1] * self.half_width
        self.x = vect_u[0]*grid_u + vect_v[0]*grid_v + self.pos[0]
        self.y = vect_u[1]*grid_u + vect_v[1]*grid_v + self.pos[1]
        self.z = vect_u[2]*grid_u + vect_v[2]*grid_v + self.pos[2]
        
    def normal(self, pos=None):
        return self.normal_static
    
    def pw_dist(self, part, p):
        n = self.normal_static
        dx = part.pos[p] - self.pos
        gap = dx.dot(n) - part.radius[p]        
        return gap
  
    def pw_col_times(self, part, pw_col_mask):
        n = self.normal_static
        dx = part.pos - self.pos
        b = dx.dot(n) - part.radius
        a = -1*part.vel.dot(n)
        t_pw = solve_linear(a,b) # defined in helper
        t_pw[pw_col_mask] = np.inf # If just hit, ignore
        t_pw[t_pw<0] = np.inf # Ignore negative times
        return t_pw
    
    
class ConeWall(WallClass):
    def __init__(self, pos=[0,0,0], axis=[0,0,1], name='cone', height=np.inf, vertex_angle=np.pi/3, collision_law='specular'):
        self.name = name
        self.dim = 3
        self.pos = np.asarray(pos).astype(float)
        self.axis = munit(axis).astype(float) # defined in helper
        self.height = height
        self.basis = make_orth_norm_basis(self.axis) # defined in helper
        self.basis[[0,-1]] = self.basis[[-1,0]]
        self.basis = self.basis.T  # basis vectors now in written down columns, axis rightmost
        self.vertex_angle = vertex_angle
        self.sinva = np.sin(vertex_angle)
        self.cosva = np.cos(vertex_angle)
        self.tanva = np.tan(vertex_angle)        
        self.collision_law = collision_law
        
    def parametrize(self):
        grid_theta = np.linspace(0, 2 * np.pi, 20)
        grid_u = np.linspace(0, self.height, 10)        
        grid_theta, grid_u = np.meshgrid(grid_theta, grid_u)
        self.x = grid_u * self.tanva * np.cos(grid_theta)
        self.y = grid_u * self.tanva * np.sin(grid_theta)
        self.z = grid_u

        c = np.array([self.x, self.y, self.z])
        c = np.rollaxis(c,0,-1)
        self.x, self.y, self.z = lab_frame.dot(c) + self.pos[:,np.newaxis,np.newaxis]


    def normal(self, pos=[0,0,20]):
        pos = np.asarray(pos)
        if pos.ndim > 1:
            raise Exception('I produce normals for the cone one point at a time.')
        dx = pos - self.pos
        px, py, pz = (dx.dot(self.basis)).T # change to cone coordinates, z is along axis
        sx = abs(px) < 1e-4
        sy = abs(px) < 1e-4
        if sx & sy:
            nx = 0
            ny = 0
            nz = 1
        elif sx:
            nx = 0
            ny = -1 * self.cosva
            nz = self.sinva
        elif sy:
            nx = -1 * self.cosva
            ny = 0
            nz = self.sinva
        else:
            w = py / px
            nx = -1 * self.cosva * np.sign(px) / np.sqrt(1+w**2)
            ny = -1 * self.cosva * np.sign(py) / np.sqrt(1+(1/w)**2)
            nz = self.sinva
        n = np.array([nx,ny,nz])
        return (self.basis).dot(n)
    
    def pw_dist(self, part, p):
        dx = part.pos[p] - self.pos
        n = self.normal(part.pos[p])
        gap = dx.dot(n) - part.radius[p]        
        return gap

    def pw_col_times(self, part, pw_col_mask):
        dx = part.pos - self.pos
        px, py, pz = (dx.dot(self.basis)).T
        vx, vy, vz = ((part.vel).dot(self.basis)).T
        k = self.tanva**2
        h = part.radius / self.sinva
        a = vx**2 + vy**2 - k*vz**2
        b = 2*px*vx + 2*py*vy - 2*k*pz*vz + 2*k*h*vz
        c = px**2 + py**2 - k*pz**2 + 2*k*h*pz - k*h**2
        t_small, t_big = solve_quadratic(a, b, c)
        t_small[pw_col_mask] = np.inf # If just hit, ignore smallest time (in absolute value)
        t_small[t_small<0] = np.inf #Ignore negative times
        t_big[t_big<0] = np.inf #Ignore negative times
        t_pw = np.fmin(t_small, t_big)
        return t_pw
    
    
class Particles():
    def __init__(self, wall, num=5, mass=3, gamma='uniform', radius=2, temp=10, vel=None, pos=None, spin=None, rot=None, collision_law='specular'):
        self.dim = wall[0].dim
        self.num = num
        self.collision_law = collision_law
        self.set_param('mass', mass, ndim=0)
        self.set_param('radius', radius, ndim=0)
        self.set_param('temp', temp, ndim=0)
        
        if(gamma == 'uniform'):
            gamma = np.sqrt(2/(2+self.dim))
        elif(gamma == 'shell'):
            gamma = np.sqrt(2/self.dim)
        else:
            gamma = np.asarry(gamma)
            if np.any((gamma < 0) | (gamma > np.sqrt(2/dim))):
                raise Exception('illegal mass distribution parameter {}'.format(gamma))
        self.set_param('gamma', gamma, ndim=0)
        self.mom_inert = self.mass * (self.gamma * self.radius)**2

        self.set_init_vel(vel)
        self.set_init_pos(pos)        
        self.set_init_spin(spin)
        self.set_init_rot(rot)        
        
    def set_param(self, key, val, ndim=0):
        msg = 'Trying to initialize {}.'.format(key)
        success = False
        target_sh = [self.dim]*ndim
        target_sh[0:0] = [self.num]
        msg = msg + '  Shape should be {}.'.format(target_sh)
        val = np.asarray(val).astype(float)
        if val.ndim == 0:
            val = np.tile(val, self.num)
            msg = msg + '  Scalar detected.  Expanding first dim to num particles={}.'.format(self.num)
        for _ in range(ndim-(val.ndim-1)):
            msg = msg + '  Apending new dimension and tiling to length dim={}.'.format(self.sim)
            val = np.tile(val[...,np.newaxis], self.dim)
        success = np.array_equal(val.shape, target_sh)
        if success:
            setattr(self, key, val)
            msg = msg + '  Success!!\n'
        else:
            msg = msg + '  Failed!!\n'
            raise Exception(msg)
        #print(msg)
        return success
    
       
    def set_init_vel(self, vel=None):
        vel = np.asarray(vel)
        self.vel = np.full([self.num,self.dim], np.inf)
        for p in range(self.num):
            try:
                self.vel[p] = vel[p]
            except:
                sigma = np.sqrt(BOLTZ_CONST * self.temp[p] / self.mass[p])
                self.vel[p] = np.random.normal(0,sigma,size=[self.dim])
        if np.isinf(self.vel).any():
            raise Exception('Could not initialize velocity')
        
    def set_init_rot(self, rot=None):
        rot = np.asarray(rot)
        self.rot = np.full([self.num,self.dim,self.dim], np.inf)
        for p in range(self.num):            
            try:
                self.rot[p] = rot[p]
            except:
                self.rot[p] = make_skew_symmetric(np.ones([self.dim,self.dim])).astype(float)
        if np.isinf(self.rot).any():
            raise Exception('Could not initialize rotation')

    def set_init_spin(self, spin=None):
        spin = np.asarray(spin)
        self.spin = np.full([self.num,self.dim,self.dim], np.inf)
        for p in range(self.num):            
            try:
                self.spin[p] = spin[p]
            except:
                sigma = np.sqrt(BOLTZ_CONST * self.temp[p] / self.mom_inert[p])
                self.spin[p] = make_skew_symmetric(np.random.normal(0,sigma,size=[self.dim,self.dim]))
        if np.isinf(self.spin).any():
            raise Exception('Could not initialize spin')


    def set_init_pos(self, pos=None):
        pos = np.asarray(pos)
        self.pos = np.full([self.num,self.dim], np.inf)
        for p in range(self.num):
            try:
                guess = pos[p]
            except:
                guess = None
            success = self.randomize_pos(p, guess=guess)
        success = self.check_positions()
        if not success:
            raise Exception('Could not initialize position')
                        
    def randomize_pos(self, p, guess=None):
        r = self.radius[p]
        success = False
        try:
            if len(guess) == self.dim:
                self.pos[p] = guess
                success = self.check_positions(p)
        except:
            pass
        max_attempts = 50
        attempt = 0
        while success == False:
            if attempt >= max_attempts:
                raise Exception('Could not place particle {}'.format(p))
            c = np.array([np.random.uniform(bb[0]+r, bb[1]-r) for bb in bounding_box])
            self.pos[p] = lab_frame.dot(c)
            #with warnings.catch_warnings():
            #    warnings.simplefilter('ignore', RuntimeWarning)
            success = self.check_positions(p)
            attempt += 1
        return success


    def pp_dist(self, p=None):
        if p is None:
            p = self.num
        dx = cross_subtract(self.pos[:p], self.pos[:p])
        r =  cross_subtract(self.radius[:p], -1*self.radius[:p])
        gap = np.sqrt(np.einsum('pqd, pqd -> pq', dx, dx)) - r
        np.fill_diagonal(gap, np.inf)
        return gap
    
    def check_positions(self, p=None):
        tol = -1e-4
        pp_dist = self.pp_dist(p)
        pp_ok =  pp_dist > tol
        
        try:
            pw_dist = np.array([[w.pw_dist(self, p) for w in wall]])
        except:            
            pw_dist = np.array([[w.pw_dist(self, p) for w in wall] for p in range(self.num)])            
        pw_ok = pw_dist > tol
        
        success = np.all(pp_ok) & np.all(pw_ok)
        return success

    def pp_col_times(self, pp_col_mask):
        dx = cross_subtract(self.pos, self.pos)
        dv = cross_subtract(self.vel)
        r =  cross_subtract(self.radius, -1*self.radius)
        a = np.einsum('pqd, pqd -> pq', dv, dv)
        b = np.einsum('pqd, pqd -> pq', dx, dv) * 2
        c = np.einsum('pqd, pqd -> pq', dx, dx) - r**2
        ## More readable, but slower version
#         a = (dv*dv).sum(axis=-1)
#         b = (dx*dv).sum(axis=-1)
#         c = (dx*dx).sum(axis=-1) - r**2
        t_small, t_big = solve_quadratic(a, b, c)
        t_small[pp_col_mask] = np.inf # If just hit, ignore smallest time (in absolute value)
        t_small[t_small<0] = np.inf #Ignore negative times
        t_big[t_big<0] = np.inf #Ignore negative times
        t_pp = np.fmin(t_small, t_big)
        return t_pp
    
    def pp_specular_law(self, p1, p2):
        """
        Particle-particle no slip law in any dimension, per Renato Feres.
        """
        m1 = self.mass[p1]
        m2 = self.mass[p2]
        M = m2 + m1
        nu = munit(self.pos[p2] - self.pos[p1])
        dv = self.vel[p2] - self.vel[p1]
        w = Pi_nu(dv, nu)
        self.vel[p1] += 2 * (m2/M) * w
        self.vel[p2] -= 2 * (m1/M) * w

    def pp_no_slip_law(self, p1, p2):
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        r1 = part.radius[p1]
        r2 = part.radius[p2]        

        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        dx = part.pos[p2] - part.pos[p1]    
        nu = munit(dx)
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
        
    def resolve_collision(self, p1, p2):
        if self.collision_law == 'specular':
            self.pp_specular_law(p1, p2)
        elif self.collision_law == 'no_slip':
            self.pp_no_slip_law(p1, p2)
        else:
            raise Exception('Unknown pp collision law {}'.format(self.collision_law))
            

def init(wall, part):
    dim = part.dim
    for w in wall:
        if ((w.dim != dim) | (len(w.pos) != dim)):
            raise Exception('Wall dim mismatch')
    if ((part.dim != dim) | (part.pos.shape[1] != dim) | (part.vel.shape[1] != dim) | (part.rot.shape[1] != dim) | (part.rot.shape[2] != dim) | (part.spin.shape[1] != dim) | (part.spin.shape[2] != dim)):
        raise Exception('Particle dim mismatch')

    global t, t_hist, col_type_hist, pos_hist, vel_hist, rot_hist, spin_hist, pp_col_mask, pw_col_mask
    t = 0
    t_hist = np.zeros(max_steps+1)
    col_type_hist = []

    pos_hist = np.zeros(np.insert(part.pos.shape, 0, max_steps+1))
    vel_hist = np.zeros(np.insert(part.vel.shape, 0, max_steps+1))
    rot_hist = np.zeros(np.insert(part.rot.shape, 0, max_steps+1))
    spin_hist = np.zeros(np.insert(part.spin.shape, 0, max_steps+1))

    pos_hist[0] = part.pos.copy()
    vel_hist[0] = part.vel.copy()
    rot_hist[0] = part.rot.copy()
    spin_hist[0] = part.spin.copy()

    pp_col_mask = np.full([part.num, part.num], False, dtype='bool')
    pw_col_mask = np.full([part.num, len(wall)], False, dtype='bool')

            
def do_the_evolution():
    global wall, part, t, pp_col_mask, pw_col_mask
    global t_hist, col_type_hist, pos_hist, vel_hist, rot_hist, spin_hist
        
    dt_pp = part.pp_col_times(pp_col_mask)
    dt_pw = np.array([w.pw_col_times(part, pw_col_mask[:,i]) for (i,w) in enumerate(wall)]).T
    #print("##########################")
    #print("STEP %d"%step)
    #display(dt_pp)
    #display(dt_pw)
       
    dt = min(dt_pp.min(), dt_pw.min())        
    #print(dt)

        
    tol = 1e-6
    pp_col_mask = (dt_pp-dt) < tol
    pp_counts = pp_col_mask.sum(axis=1)
    pw_col_mask = (dt_pw-dt) < tol
    pw_counts = pw_col_mask.sum(axis=1)
            
    #display(pp_col_mask)
    #display(pw_col_mask)
    #display(part.pos)
    #display(part.vel)
    
 
    cmplx_A = (pp_counts >= 2)
    cmplx_B = (pp_counts >= 1) & (pw_counts >= 1)
    cmplx_mask = cmplx_A | cmplx_B
    
    t += dt
    part.pos += part.vel * dt
    ### We don't need to compute the rotational orientation at this point.  It does not affect collision points, times, or post-collision velocities.  We'll do this later when we aniamte.
    # for p in range(part.num):
    #     drot = linalg.expm(part.spin[p] * dt)        
    #     part.rot[p] = part.rot[p].dot(drot)

    col_types = []
    for p in np.nonzero(cmplx_mask)[0]:
        part.randomize_pos(p)
        pp_col_mask[p,:] = False
        pp_col_mask[:,p] = False
        pw_col_mask[p,:] = False
        pp_counts = pp_col_mask.sum(axis=1)
        pw_counts = pw_col_mask.sum(axis=1)
        
    for (p1, p2) in zip(*np.nonzero(pp_col_mask)):
        if p1 < p2:
            col_types.append('p{}p{}'.format(p1,p2))
            part.resolve_collision(p1, p2)
            

    for p in np.nonzero(pw_counts)[0]:
        ws = np.nonzero(pw_col_mask[p])[0]
        col_types.append('p{}w{}'.format(p,*ws))
        if len(ws) == 1: # single wall collision
            w = ws[0]
            wall[w].resolve_collision(part, p)

        else: # multiple walls (corner collision)
            def vel_normal():
                '''
                velocity normal to each wall, negative<->toward, positive<->away
                '''
                return np.array([part.vel[p].dot(wall[w].normal(part.pos[p]))  for w in ws])

            msg = "\nCorner collision at step {} of particle {} at walls {}.  Vel in = {}\n".format(step, p, ws, part.vel[p])
            # order walls by particle normal velocity
            o = vel_normal()
            srt = np.argsort(o) # sort the velocities, most negative first
            ws = ws[srt] # order walls in the same order
            # apply collision at each wall at least once
            for w in ws:
                wall[w].resolve_collision(part, p)
                msg = msg + "Applying {} type collision at wall {}.  New velocity = {}\n".format(wall[w].collision_law,w,part.vel[p])
            
            # due to interaction among the wall collision laws, particle's new velocity
            # may take it out of the chamber.  Check for this and apply the
            # SPECULAR law for the worst wall
            max_attempts = 2*len(ws)
            for attempt in range(max_attempts):
                o = vel_normal()
                msg = msg + 'Normal velocities are now {}\n'.format(vel_normal())
                success = np.all(o >= 0)
                if success == True:
                    msg = msg + 'Done'
                    break
                else:                    
                    w = ws[o.argmin()]
                    wall[w].pw_specular_law(part, p)
                    msg = msg + "Applying specular law at wall {}. New velocity = {}\n".format(w, part.vel[p])
                    
            # If particle is STILL moving toward outside of the chamber, re-randomize its position
            if success == False:
                part.randomize_pos(p)
            #print(msg)
            
        success = part.check_positions()
        if not success:
            raise Exception(' Failure at step %d'%step)

    
    t_hist[step+1] = t
    col_type_hist.append(col_types)
                        
                    
    pos_hist[step+1] = part.pos.copy()
    vel_hist[step+1] = part.vel.copy()
    rot_hist[step+1] = part.rot.copy()
    spin_hist[step+1] = part.spin.copy()