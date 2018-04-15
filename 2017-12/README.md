## This is an older version with gpu support.  Recommend new version.


[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/drscook/no_slip_billiards/master)

# no_slip_billiards

This program simulates the no-slip billiard dynamical systems introduced by Groomhead and Gutkin and extended by Chris Cox and Renato Feres.  See:

- http://www.sciencedirect.com/science/article/pii/016727899390205F?via%3Dihub
- https://arxiv.org/abs/1602.01490
- https://arxiv.org/abs/1612.03355

This simulation should work in any dimension, but has been tested in 2D and 3D.  Note that all particles are assumed to be spherical, but their mass distribution can be controlled via the gamma parameter.

Dependencies
- Python 3 (written and tested with 3.6, but should be backward compatible)
- scientific stack (numpy, matplotlib, etc)
- plotly (for interactive graphics)
- FFMEG or equivalent (for creating video files)

Future developments
- Add cylinder and sphere shaped boundary pieces
- Add random billiard collision laws (both thermally passive and thermally active)
- Improve 3D animations to show spinning body like 2D animations do now.  (3D animations currently only show path of particle centers)
- Video files for 3D animations
- Develop interactive widget to change physical parameters and re-run experiment from a GUI
- Write data to file for future analysis

Code comments:
- Each different type of experiment should get its own Jupyter notebook which calls the stuff in the programs folder
- Not everything in helper.py is relevant for this project.  It is my collection of convenience functions I share across many projects.
