### common chamber geometries

def box(cell_size):

    Tangents = np.diag(cell_size)
    wall = []
    for d in range(dim):
        for s in [-1,1]:
            v = np.zeros(dim, dtype=np_dtype)
            v[d] = s*cell_size[d]
            wall.append(FlatWall(base_point=v.copy(), normal=-v.copy()
                                   ,tangents = np.delete(Tangents,d,0)))
    return wall

def sinai(cell_size, scatter_radius):    
    if np.any(scatter_radius > cell_size):
        raise Exception('scatterer largers than box')
    wall = box(cell_size)
    wall.append(SphereWall(base_point=np.zeros(dim), radius=scatter_radius))
    return wall
    
def lorentz_rectangle(cell_size, scatter_radius):
    wall = sinai(cell_size, scatter_radius)
    s = -1
    for (k,w) in enumerate(wall[:-1]):
        w.gap_m = 0.0
        w.gap_b = 0.0
        d = int(np.floor(k/2))
        s *= -1
        w.collision_law = PW_PeriodicLaw(wrap_dim=d, wrap_wall=k+s)
    return wall

def lorentz_hexagonal(scatter_radius, part_radius, horizon_factor):
    # horizon_factor < 1 for finite horizon, horizon_factor > 1 for infinite horizon
    R = scatter_radius + part_radius
    gap_crit = (2/np.sqrt(3) - 1) * R
    gap = horizon_factor * gap_crit
    x0 = R + gap
    y0 = np.sqrt(3) * x0
    cell_size = np.array([x0,y0])
    
    wall = lorentz_rectangle(cell_size, scatter_radius)
    wall.append(SphereWall(base_point=np.array([x0,y0]), radius=scatter_radius))
    wall.append(SphereWall(base_point=np.array([-x0,y0]), radius=scatter_radius))
    wall.append(SphereWall(base_point=np.array([-x0,-y0]), radius=scatter_radius))
    wall.append(SphereWall(base_point=np.array([x0,-y0]), radius=scatter_radius))
    return wall, cell_size