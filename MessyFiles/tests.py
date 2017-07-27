# exec(open("C:/PythonStuff/tests.py").read())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# layout
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(left=0.20, bottom=0.30)

# where sliders are
axcolor = 'lightgoldenrodyellow'
phase1_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
noise1_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
noise2_ax = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

# data
t = np.arange(0.0, 1.0, 0.01)
f = 2
p1 = 2

noise1 = np.random.normal(0, 1, 100)
noise2 = np.random.normal(0, 1, 100)

s1 = np.sin(2*np.pi*f*t + p1)
s2 = np.sin(2*np.pi*f*t)



# plots
plt.subplot(121)
l1, = plt.plot(t, s1, marker="o", markersize=4, label="signal1", lw=1, color='blue')
l2, = plt.plot(t, s2, marker="o", markersize=4, label="signal2", lw=1, color='orange')
plt.subplot(122)
l3, = plt.plot(s1, s2, marker="o", markersize=4, lw=0)



phase1_slide = Slider(phase1_ax, "phase1", 0, 10)
noise1_slide = Slider(noise1_ax , "noise1", 0, 2, valinit=0)
noise2_slide = Slider(noise2_ax , "noise2", 0, 2, valinit=0)


def update(val):
	print(noise1_slide.val)
	p1 = -phase1_slide.val
	l1.set_ydata(np.sin(2*np.pi*f*t + p1) + noise1_slide.val*noise1)
	l3.set_xdata(np.sin(2*np.pi*f*t + p1) + noise1_slide.val*noise1)
	fig.canvas.draw_idle()


# sfreq.on_changed(update)
# samp.on_changed(update)
phase1_slide.on_changed(update)
noise1_slide.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


# def reset(event):
#     sfreq.reset()
#     samp.reset()
# button.on_clicked(reset)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()
# radio.on_clicked(colorfunc)

plt.show()
