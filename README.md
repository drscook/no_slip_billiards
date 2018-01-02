[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/drscook/no_slip_billiards/mybinder_friendly)

# no_slip_billiards

This program simulates the no-slip billiard dynamical systems introduced by Groomhead and Gutkin and extended by Chris Cox and Renato Feres

- http://www.sciencedirect.com/science/article/pii/016727899390205F?via%3Dihub
- https://arxiv.org/abs/1602.01490
- https://arxiv.org/abs/1612.03355

This simulation should work in any dimension, but has been tested in 2D and 3D.  Note that all particles are assumed to be spherical, but their mass distribution can be controlled via the gamma parameter.

Dependencies
- Python 3 (written and tested with 3.6, but should be backward compatible)
- scientific stack (numpy, matplotlib, etc)
- FFMEG for creating video files (optional)
- plotly and bqplot (optional - for interactive graphics)

Future developments
- Improve 3D animations to show spinning body like 2D animations do now.  (3D animations currently only show path of particle centers)
- Add cylinder and sphere shaped boundary pieces
- Add random billiard collision laws (both thermally passive and thermally active)
- Develop interactive widget to run entire experiments from a GUI
- Write data to file for future analysis
- Explore Binder or other platforms so users can use without installing python


Code comments:
- Each different type of experiment should get its own Jupyter notebook which calls the stuff in the programs folder
- Not everything in helper.py is relevant for this project.  It is my collection of convenience functions I share across many projects.
