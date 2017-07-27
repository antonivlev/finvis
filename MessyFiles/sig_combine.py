# exec(open("C:/PythonStuff/sig_combine.py").read())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# layout
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.subplots_adjust(left=0.20, bottom=0.30)
axcolor = 'lightgoldenrodyellow'

# where sliders are
c1_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
c2_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
a1_ax = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
a2_ax = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

# data
t = np.arange(-10, 10, 1)
a1 = 1
c1 = 0
a2 = 1
c2 = 1

def sig(tvals, amp, c):
	sig_vals = []
	for t in tvals:
		if abs(t - c) > 0:
			sig_vals.append(0)
		else:
			sig_vals.append(1 - (1/amp)*abs(t - c))
	return np.array(sig_vals)


s1 = sig(t, a1, c1)
s2 = sig(t, a2, c2)

# plots
plt.subplot(111)
l1, = plt.plot(t, s1, label="signal1", lw=1, color='blue')
l2, = plt.plot(t, s2, label="signal2", lw=1, color='orange')
l3, = plt.plot(t, s1+s2, label="signal2", lw=0.5, color='red')

a1_slide = Slider(a1_ax, "a1", 1, 10)


def update(val):
	a1 = a1_slide.val
	s1 = sig(t, a1, c1)
	print(s1)
	l1.set_ydata(s1)
	l3.set_ydata(s1+s2)
	fig.canvas.draw_idle()

a1_slide.on_changed(update)

plt.show()
