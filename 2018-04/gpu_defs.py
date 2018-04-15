### Code for parallel mode on GPU using numba cuda

import math
import numba as nb
import numba.cuda as cuda
nb_dtype = nb.float64

num_walls = len(wall)
num_part = part.num
pp_bcols = min(num_part, sqrt_threads_per_block_max)
pp_brows = pp_bcols
pp_gcols = int(np.ceil(num_part / pp_bcols))
pp_grows = pp_gcols
pp_block_shape = (pp_bcols, pp_brows)
pp_grid_shape = (pp_gcols, pp_grows)
assert pp_block_shape[1] * pp_grid_shape[1] >= num_part
assert pp_block_shape[0] == pp_block_shape[1]
assert pp_grid_shape[0] == pp_grid_shape[1]

pw_bcols = num_walls
pw_brows = int(np.floor(threads_per_block_max / pw_bcols))
pw_gcols = 1
pw_grows = int(np.ceil(num_part / pw_brows))
pw_block_shape = (pw_bcols, pw_brows)
pw_grid_shape = (pw_gcols, pw_grows)
assert pw_block_shape[1] * pw_grid_shape[1] >= num_part
assert pw_block_shape[0] * pw_grid_shape[0] >= num_walls

def disp_gpu(A):
    B = A.get('host')
    C = A.get('gpu').copy_to_host()
    print('host')    
    print(B)
    print('device')    
    print(C)
    assert np.allclose(B,C)


def load_gpu(part):
    global mode
    
    if part.num < part.dim:
        print('Can only use parallel processing when {} = part_num  >= dim = {}. Switching to serial.'.format(part.num, part.dim))
        mode = 'serial'
        return

    r = part.num % sqrt_threads_per_block_max
    if r != 0:
        print('I urge you to set num_part to a multiple of sqrt_threads_per_block_max = {}'.format(sqrt_threads_per_block_max))
        
    print('load_gpu')
    
    part.radius_gpu = cuda.to_device(part.radius)

    part.pos_smrt = nb.SmartArray(part.pos)
    part.pos = part.pos_smrt.get('host')

    part.vel_smrt = nb.SmartArray(part.vel)
    part.vel = part.vel_smrt.get('host')

    part.pw_mask_smrt = nb.SmartArray(part.pw_mask)
    part.pw_mask = part.pw_mask_smrt.get('host')

    part.pp_mask_smrt = nb.SmartArray(part.pp_mask)
    part.pp_mask = part.pp_mask_smrt.get('host')
    
    part.pp_dt_block = np.full([part.num, pp_grid_shape[0]], np.inf, dtype=np_dtype)
    part.pp_dt_block_smrt = nb.SmartArray(part.pp_dt_block)
    part.pp_dt_block = part.pp_dt_block_smrt.get('host')

#     part.wall_base_point_gpu = cuda.to_device(np.vstack([w.base_point for w in wall]))
#     part.wall_normal_gpu = cuda.to_device(np.vstack([w.normal for w in wall]))
#     part.pw_gap_min_gpu = cuda.to_device(np.vstack([w.pw_gap_min for w in wall]))

    update_gpu(part)
    check_gpu_sync()
    


def is_synced(cpu, smrt):
    assert np.allclose(cpu, smrt.get('host'), rtol=rel_tol)
    assert np.allclose(cpu, smrt.get('gpu'), rtol=rel_tol)

def check_gpu_sync():
    is_synced(part.pos, part.pos_smrt)
    is_synced(part.vel, part.vel_smrt)
    is_synced(part.pw_mask, part.pw_mask_smrt)
    is_synced(part.pp_mask, part.pp_mask_smrt)

def update_gpu(part):
    part.pos_smrt.mark_changed('host')
    part.vel_smrt.mark_changed('host')
    part.pw_mask_smrt.mark_changed('host')
    part.pp_mask_smrt.mark_changed('host')
    
def get_pp_col_time_gpu(part):
    global errors
    check_gpu_sync()
    get_pp_col_time_kernel[pp_grid_shape, pp_block_shape](part.pp_dt_block_smrt, part.pos_smrt, part.vel_smrt, part.pp_mask_smrt, part.num, part.radius_gpu)#, part.pp_dt_full_gpu, part.pp_a_gpu, part.pp_b_gpu, part.pp_c_gpu)

#     part.pp_dt_block_smrt.mark_changed('gpu')
#     part.pp_dt_gpu = np.min(part.pp_dt_block_smrt.get('gpu'), axis=1)

    part.pp_dt_gpu = part.pp_dt_block_smrt.min(axis=1)
    
    if check_gpu_against_cpu == True:
#         print('checking against cpu')
        part.get_pp_col_time_cpu()
        if not np.allclose(part.pp_dt, part.pp_dt_gpu, rtol=rel_tol):
            print('pp_dt_cpu and pp_dt_gpu do not match')
            D = part.pp_dt_gpu - part.pp_dt
            idx = np.nanargmax(D)
            print('index {} if off by {}'.format(idx,D[idx]))
            errors +=1
            print('errors = {}'.format(errors))
    else:
        part.pp_dt = part.pp_dt_gpu.copy()

    
@cuda.jit
def get_pp_col_time_kernel(pp_dt, pos, vel,  mask, N, radius):#, pp_dt_full, a_gpu, b_gpu, c_gpu):
    pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    p = ty + cuda.blockIdx.y * cuda.blockDim.y
    q = tx + cuda.blockIdx.x * cuda.blockDim.x
    if ((p >= N) or (q >= N)):        
        pp_dt_shr[ty,tx] = np.inf
        a = p
        b = q
        c = N
    else:
        a = 0.0
        b = 0.0
        c = 0.0
        for d in range(dim):
            dx = pos[p,d] - pos[q,d]
            dv = vel[p,d] - vel[q,d]
            a += (dv * dv)
            b += (dx * dv * 2)
            c += (dx * dx)
        c -= (radius[p] + radius[q])**2

        if ((mask[0]==p) & (mask[1]==q)):
            masked = True
        elif ((mask[0]==q) & (mask[1]==p)):
            masked = True
        else:
            masked = False

        pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)

    cuda.syncthreads()

    row_min_gpu(pp_dt_shr)
    pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty, 0]
    cuda.syncthreads()

#         a_gpu[p,q] = pos[p,0]
#         b_gpu[p,q] = pos[p,1]
#         c_gpu[p,q] = pos[p,2]
#         pp_dt_full[p,q] = pp_dt_shr[ty,tx]

    
        
@cuda.jit(device=True)
def solve_quadratic_gpu(a, b, c, mask=False):
    # The hard task is finding the first root.  Because the sum and product of the two roots must equal
    # -b/a and c/a resp, we can compute the second easily.
    # We use a combination of the Citardauq and the quadratic formulas.  Each is numerically stable for a
    # different range of coeficients.

    M = 1e8
    small = np.inf
    big = np.inf
    d = b**2 - 4*a*c
    if d >= 0:
        d = math.sqrt(d)        
        if b < 0:
            d *= -1
            
        e = -(b + d) / 2
        if abs(c) < M * abs(e):
            small = c / e
        else:
            f = -(b - d) / 2
            if abs(f) < M * abs(a):
                small = f / a
                
        if abs(b) < M * abs(a):
            big = - b / a - small
        else:
            g = a * small
            if abs(c) < M * abs(g):
                big = c / g

        if mask == True:
            small = big
#             big = np.inf
        if small < 0:
            if big < 0:
                small = np.inf                
            else:
                small = big
#             big = np.inf
#         elif big < 0:
#             big = np.inf
    return small



@cuda.jit(device=True)
def row_min_gpu(A):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    m = float(cuda.blockDim.x)
    while m > 1:
        n = m / 2
        k = int(math.ceil(n))
        if (tx + k) < m:
            if A[ty,tx] > A[ty,tx+k]:
                A[ty,tx] = A[ty,tx+k]
        m = n
        cuda.syncthreads()