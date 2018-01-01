# no_slip_billiards

This program simulates the no-slip billiard dynamical systems introduced by Groomhead and Gutkin and extended by Chris Cox and Renato Feres

http://www.sciencedirect.com/science/article/pii/016727899390205F?via%3Dihub
https://arxiv.org/abs/1602.01490
https://arxiv.org/abs/1612.03355

This simulation should work in any dimension, but has been tested in 2 and 3D.  Note that all particles are assumed to be spherical, but their mass distribution can be controlled via the gamma parameter.

Dependencies
Python 3 (written and tested with 3.6, but should be backward compatible)
Python scientific stack (numpy, matplotlib, etc)
FFMEG for creating video files (optional)
plotly and bqplot (optional - for interactive graphics)
