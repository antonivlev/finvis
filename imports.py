''' Include this where you need these:
exec(open("C:/PythonStuff/imports.py").read())

start jupyter:
jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000

'''

import matplotlib.pyplot as plt, mpld3
import numpy as np
import random

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
plotly.offline.init_notebook_mode(connected=True)

from numpy.fft import fft
from numpy import pi, sin, cos
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA