## Code for visualizations


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import ipywidgets as widgets
import itertools as it
import scipy.linalg
import io
import base64
rc('animation', html='jshtml')


def draw(s=0):
    get_mesh()
    s %= part.num_frames

    if hasattr(part, 're_pos'):
        x = part.re_pos[:s+1]
    else:
        x = part.pos_hist[:s+1]

    if hasattr(part, 're_orient'):
        o = part.re_orient[s]
    else:
        o = part.orient

    translates = get_cell_translates(x)

    fig, ax = plt.subplots(figsize=[8,8]);
    for trans in translates:
        for w in wall:
            ax.plot(*((w.mesh+trans).T), color='black')
            
    for p in range(part.num):
        ax.plot(*(x[:,p].T), color=part.clr[p])
        ax.plot(*((part.mesh[p].dot(o[p].T) + x[-1,p]).T), color=part.clr[p])
    ax.set_aspect('equal')


def interactive_plot():
    if part.dim != 2:
        print('only works in 2D')
        return 
    
    l = widgets.Layout(width='150px')
    step_text = widgets.BoundedIntText(min=0, max=part.num_frames-1, value=0, layout=l)
    step_slider = widgets.IntSlider(min=0, max=part.num_frames-1, value=0, readout=False, continuous_update=False, layout=l)
    play_button = widgets.Play(min=0, max=part.num_frames-1, step=1, interval=500, layout=l)

    widgets.jslink((step_text, 'value'), (step_slider, 'value'))
    widgets.jslink((step_text, 'value'), (play_button, 'value'))
    
    img = widgets.interactive_output(draw, {'s':step_text})
    wid = widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img])
    display(wid)

# Animations rely on FFMPEG or equivalent.

def animate(frames=part.num_frames, run_time=5):
    get_mesh()
    frames %= part.num_frames

    x = part.re_pos[:frames+1]
    o = part.re_orient[:frames+1]

    translates = get_cell_translates(x)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for trans in translates:
        for w in wall:
            ax.plot(*((w.mesh+trans).T), color='black')

    path = []
    bdy = []
    for p in range(part.num):
        path.append(ax.plot([],[], color=part.clr[p])[0])
        bdy.append(ax.plot([],[], color=part.clr[p])[0])

    def init():
        for p in range(part.num):
            path[p].set_data([], [])
            bdy[p].set_data([], [])
        return path + bdy

    def update(s):
        for p in range(part.num):
            path[p].set_data(*(x[:s+1,p].T))
            bdy[p].set_data(*((part.mesh[p].dot(o[s,p].T) + x[s,p]).T))
        return path + bdy
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=frames, interval=run_time*1000/frames, blit=True)
    plt.close()
    return anim

def play_video(fname):
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(data='''<video alt="test" controls>
         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
         </video>'''.format(encoded.decode('ascii'))))


    
def get_cell_translates(x):
    cs = 2 * cell_size
    m = (x.min(axis=0).min(axis=0) / cs).round()
    M = (x.max(axis=0).max(axis=0) / cs).round()
    z = [np.arange(m[d],M[d]+1)*cs[d] for d in range(part.dim)]
    translates = it.product(*z)
    return tuple(translates)

def get_mesh():
    for w in wall:
        if hasattr(w, 'mesh') == False:
            w.get_mesh()

    if hasattr(part, 'mesh') == False:
        part.get_mesh()

def smoother(part, min_frames=None, orient=True):
    print('smoothing')
    t, x, v, s = part.t_hist, part.pos_hist, part.vel_hist, part.spin_hist
    dts = np.diff(t)
    if (min_frames is None):
        ddts = dts
        num_frames = np.ones(dts.shape, dtype=int)
    else:
        short_step = dts < abs_tol
        nominal_frame_length = np.percentile(dts[~short_step], 25) / min_frames
        num_frames = np.round(dts / nominal_frame_length).astype(int) # Divide each step into pieces of length as close to nominal_frame_length as possible
        num_frames[num_frames<1] = 1
        ddts = dts / num_frames  # Compute frame length within each step

    # Now interpolate.  re_x denotes the interpolated version of x
    re_t, re_x, re_v, re_s = [t[0]], [x[0]], [v[0]], [s[0]]
    re_o = [part.orient]
    for (i, ddt) in enumerate(ddts):
        re_t[-1] = t[i]
        re_x[-1] = x[i]
        re_v[-1] = v[i]
        re_s[-1] = s[i]
        dx = re_v[-1] * ddt
        if orient == True:
            do = [scipy.linalg.expm(ddt * U) for U in re_s[-1]] # incremental rotatation during each frame

        for f in range(num_frames[i]):
            re_t.append(re_t[-1] + ddt)
            re_x.append(re_x[-1] + dx)
            re_v.append(re_v[-1])
            re_s.append(re_s[-1])
            if orient == True:
                #B = [A.dot(Z) for (A,Z) in zip(re_o[-1], do)] # rotates each particle the right amount
                B = np.einsum('pde,pef->pdf', re_o[-1], do)
                re_o.append(np.array(B))
            else:
                re_o.append(re_o[-1])

    part.re_t = np.asarray(re_t)
    part.re_pos = np.asarray(re_x)
    part.re_vel = np.asarray(re_v)
    part.re_orient = np.asarray(re_o)
    part.re_spin = np.asarray(re_s)
    part.num_frames = len(part.re_t)

def flat_mesh(tangents):
    pts = 100
    N, D = tangents.shape
    grid = [np.linspace(-1, 1, pts) for n in range(N)]
    grid = np.meshgrid(*grid)
    grid = np.asarray(grid)
    mesh = grid.T.dot(tangents)
    return mesh

def sphere_mesh(dim=2, radius=1.0):
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
    return np.asarray(mesh).T
