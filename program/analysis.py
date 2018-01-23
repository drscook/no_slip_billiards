from helper import *

### To add:
# plot boundaries of particles in 3D plot
# Record 3D videos.  It works for 2D, but I have not tried to get it for 3D

from ipywidgets import interactive, interactive_output, HBox, VBox, IntSlider, BoundedIntText, Play, jslink
import scipy.linalg as la
import matplotlib.animation as animation

    
def smoother(max_distort=50):
    """
    If 0<max_distort<=100, this interpolates between collisions do give "smooth" motion for the particles.  Smaller max_distort means smoother animation but also longer processing and larger files.
    """
    global t_hist, pos_hist, vel_hist, orient_hist, spin_hist

    smooth = False
    try:
        if (max_distort > 0) and (max_distort <= 100):
            smooth = True
    except:
        pass
    
    dts = np.diff(t_hist)
    if smooth == False: 
        num_frames = np.ones_like(dts).astype(int)
        ddts = dts / num_frames
    else:
        # We will divide the time between each pair of successive collisions into frames that have length as simliar as possible.
        distort = np.inf
        min_frames = 0
        
        while distort >= max_distort:
            min_frames += 1
            nominal_frame_length = dts.min() / min_frames  # Time increment = shortest time btw 2 collisions / min_frames
            num_frames = np.round(dts / nominal_frame_length).astype(int) # Divide each step into pieces of length as close to nominal_frame_length as possible
            ddts = dts / num_frames  # Compute frame length within each step
            m = ddts.mean()
            d = np.abs(ddts-m).max() # frame length farthest from average
            distort = d / m * 100 

    # Now do the interpolation.  re_x denotes the interpolated version of x
    re_t = [0]
    re_pos = [pos_hist[0]]
    re_vel = [vel_hist[0]]
    re_orient = [orient_hist[0]]
    re_spin = [spin_hist[0]]
    for (i, ddt) in enumerate(ddts):
        re_pos[-1] = pos_hist[i]
        re_vel[-1] = vel_hist[i]
        re_spin[-1] = spin_hist[i]
        dpos = re_vel[-1] * ddt
        dorient = [la.expm(s * ddt) for s in re_spin[-1]] # incremental rotatation during each frame
        # Note that orientation was not computed during the simulation because it was not needed to determine point of collision or outgoing velocity and spin.  So this is the first time orientation is computed.

        for f in range(num_frames[i]):
            re_t.append(re_t[-1] + ddt)
            re_pos.append(re_pos[-1] + dpos)
            re_vel.append(re_vel[-1])
            w = [do.dot(o) for (do,o) in zip(dorient,re_orient[-1])] # rotates each particle the right amount
            re_orient.append(np.array(w))
            re_spin.append(re_spin[-1])

    re_t = np.asarray(re_t)
    re_pos = np.asarray(re_pos)
    re_vel = np.asarray(re_vel)
    re_orient = np.asarray(re_orient)
    re_spin = np.asarray(re_spin)
    re_pos = re_pos[...,np.newaxis]  # append extra axis of length 1 for later use drawing particle boundaries.
    re_vel = re_vel[...,np.newaxis]
    
    return re_t, re_pos, re_vel, re_orient, re_spin


def compute_particle_boundaries():
    theta = np.linspace(0, 2*np.pi, 100)
    theta = np.append(theta,np.pi)
    bdy = np.array([np.cos(theta), np.sin(theta)])
    bdy = bdy[np.newaxis,...] * part.radius[:,np.newaxis,np.newaxis]
    bdy = mexpand(bdy, axis=0, length = S)  #defined in helper
    for s in range(S):
        for p in range(P):
            bdy[s,p] = re_orient[s,p].dot(bdy[s,p]) + re_pos[s,p]
    return bdy



#### incomplete & untested
def compute_particle_boundaries_3d():
    theta = np.linspace(0, 2*np.pi, 100)
    theta = np.append(theta,np.pi)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    bdy = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    bdy = bdy[np.newaxis,...] * part.radius[:,np.newaxis,np.newaxis,np.newaxis]
    bdy = mexpand(bdy, axis=0, length = S)  #defined in helper
    for s in range(S):
        for p in range(P):
            bdy[s,p] = re_orient[s,p].dot(bdy[s,p]) + re_pos[s,p]
    return bdy

def animate_2d_box(run_time=5):
    import matplotlib.pyplot as plt
    bdy = compute_particle_boundaries()
    
    x_min, x_max = bounding_box[0][0], bounding_box[0][1]
    y_min, y_max = bounding_box[1][0], bounding_box[1][1]
    
    fig = plt.figure()
    wall, = plt.plot([x_max,x_max,x_min,x_min,x_max], [y_max,y_min,y_min,y_max,y_max],'black')
    path = []
    circ = []
    time_label = fig.text(0.5, 0.84, 's=0 t=0.00', ha='center', va='top', fontsize=10)    
    
    time_label.text = ["{:.2f}".format(re_t[0])]
    for p in range(P):
        path.append(plt.plot([],[])[0])    
    plt.gca().set_prop_cycle(None) # reset colors so boundary and path are same color
    for p in range(P):
        circ.append(plt.plot([],[])[0])
    def update(s):
        time_label.set_text('s={} t={:.2f}'.format(s,re_t[s]))
        for p in range(P):
            if s == 0:
                path[p].set_data([],[])
                circ[p].set_data([],[])
            else:
                path[p].set_data(re_pos[:s+1,p,0,:], re_pos[:s+1,p,1,:])
                circ[p].set_data(bdy[s,p,0,:], bdy[s,p,1,:])
        return wall, path, circ
    anim = animation.FuncAnimation(fig, update, frames=S, interval=run_time*1000/S, blit=True)
    return anim

def interactive_plotly():
    print("using newer")
    # DO NOT run me with many steps to plot.  I will take a long time.  We should add a flag to
    # turn off animation but leave slider.  I think this should work better from large num steps.
    # I just learned how to use plotly, so I am probably not using it optimally.
    import plotly.offline as py
    import plotly.graph_objs as go
    
    py.init_notebook_mode(connected=True)
    frame_length = 200
    ease = 'linear'

    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    figure['layout']['showlegend'] = False
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': frame_length,
                'easing': 'ease'
            }
        ],
        'initialValue': '0',
        'plotlycommand': 'animate',
        'values': range(S),
        'visible': True
    }

    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': frame_length, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': frame_length, 'easing': 'ease'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 0, 't': 0},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 8},
            'prefix': 'step:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': frame_length, 'easing': 'ease'},
        'pad': {'b': 0, 't': 00},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    
    for s in range(S):
        frame = {'data': [], 'name': str(s)}
        frame['data'].extend(surface_data_dict)
        for p in range(P):
            path_data_dict = {
                'name': 'path_'+str(p),
                'mode': 'lines',
                'x': re_pos[:s+1,p,0,0],
                'y': re_pos[:s+1,p,1,0]
            }
            
            
            if wall[0].dim == 2:
                path_data_dict['type'] = 'scatter'
                bdy = compute_particle_boundaries()
                bdy_data_dict = {
                    'name': 'bdy_'+str(p),
                    'mode': 'lines',
                    'x': bdy[s,p,0,:],
                    'y': bdy[s,p,1,:]
                }
                frame['data'].append(bdy_data_dict)
            elif wall[0].dim == 3:
                path_data_dict['type'] = 'scatter3d'
                path_data_dict['z'] = re_pos[:s+1,p,2,0]
                ### add 3D boundary plot here
                #bdy_data_dict['type'] = 'scatter3d'
            frame['data'].append(path_data_dict)
            
        figure['frames'].append(frame)
        slider_step = {'args': [
            [s],
            {'frame': {'duration': 200, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 200}}
         ],
         'label': s,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    figure['data'] = figure['frames'][0]['data']
    py.iplot(figure)
    
    
def interactive_plotly_stable():
    print("using newer")
    # DO NOT run me with many steps to plot.  I will take a long time.  We should add a flag to
    # turn off animation but leave slider.  I think this should work better from large num steps.
    # I just learned how to use plotly, so I am probably not using it optimally.
    import plotly.offline as py
    import plotly.graph_objs as go
    
    py.init_notebook_mode(connected=True)
    frame_length = 200
    ease = 'linear'

    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    figure['layout']['showlegend'] = False
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': frame_length,
                'easing': 'ease'
            }
        ],
        'initialValue': '0',
        'plotlycommand': 'animate',
        'values': range(S),
        'visible': True
    }

    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': frame_length, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': frame_length, 'easing': 'ease'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 0, 't': 0},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 8},
            'prefix': 'step:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': frame_length, 'easing': 'ease'},
        'pad': {'b': 0, 't': 00},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    
    for s in range(S):
        frame = {'data': [], 'name': str(s)}
        frame['data'].extend(surface_data_dict)
        for p in range(P):
            path_data_dict = {
                'name': 'path_'+str(p),
                'mode': 'lines',
                'x': re_pos[:s+1,p,0,0],
                'y': re_pos[:s+1,p,1,0]
            }
            
            
            if wall[0].dim == 2:
                path_data_dict['type'] = 'scatter'
                bdy = compute_particle_boundaries()
                bdy_data_dict = {
                    'name': 'bdy_'+str(p),
                    'mode': 'lines',
                    'x': bdy[s,p,0,:],
                    'y': bdy[s,p,1,:]
                }
                frame['data'].append(bdy_data_dict)
            elif wall[0].dim == 3:
                path_data_dict['type'] = 'scatter3d'
                path_data_dict['z'] = re_pos[:s+1,p,2,0]
                #bdy_data_dict['type'] = 'scatter3d'
            frame['data'].append(path_data_dict)
            
        figure['frames'].append(frame)
        slider_step = {'args': [
            [s],
            {'frame': {'duration': 200, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 200}}
         ],
         'label': s,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    figure['data'] = figure['frames'][0]['data']
    py.iplot(figure)
    

## older function for interactive graphics that relies on bqplot.  Does not work well on mybinder, so replaced by plotly.
#def interactive_2d_box(time_interval=1):
#     from bqplot import pyplot as plt
#     bdy = compute_particle_boundaries()

#     x_min, x_max = bounding_box[0][0], bounding_box[0][1]
#     y_min, y_max = bounding_box[1][0], bounding_box[1][1]
    
#     fig = plt.figure(min_aspect_ratio=0.9,max_aspect_ratio=1.1,animation_duration=time_interval)
#     wall = plt.plot([x_max,x_max,x_min,x_min,x_max], [y_max,y_min,y_min,y_max,y_max],'black')
#     path = plt.plot([],[])
#     circ = plt.plot(bdy[0,:,0,:], bdy[0,:,1,:])
#     time_label = plt.label(text='s=0 t=0.00', x=[0], y=[.9*y_max], colors=['black'], align='middle', default_size=10, font_weight='normal', enable_move=True)
#     def update(change):
#         s = step_slider.value
#         time_label.text = ['s={}\nt={:.2f}'.format(s,re_t[s])]
#         if s == 0:
#             path.x = []
#             path.y = []
#         else:
#             path.x = re_pos[:s+1,:,0,:].T
#             path.y = re_pos[:s+1,:,1,:].T
#         circ.x = bdy[s,:,0,:]
#         circ.y = bdy[s,:,1,:]
#     step_slider = IntSlider(min=0, max=S, step=1, description='step', value=0, continuous_update=False)
#     play_button = Play(min=0, max=S, step=1, interval=time_interval)
#     step_box = BoundedIntText(min=0, max=S, description='step', value=0)
#     jslink((play_button, 'value'), (step_slider, 'value'))
#     jslink((play_button, 'value'), (step_box, 'value'))
#     step_slider.observe(update, 'value')
#     console = VBox([HBox([play_button, step_slider, step_box]), fig])
#     display(console)