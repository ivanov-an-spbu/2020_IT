import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def get_sin(x, A = 1, w=1):
    y = A*np.sin(w*x)
    return y


def sliders_on_changed(val):
    axs[0].clear()
    axs[0].plot(x, get_sin(x, A=1, w=val))
    fig.canvas.draw_idle()
    return

x = np.arange(0, 10, 0.1)

fig, axs = plt.subplots(1,2)

[line] = axs[0].plot(x, get_sin(x, 1, 0.5))


fig.subplots_adjust(left=0.25, bottom=0.35)
ax = fig.add_axes([0.25, 0.1, 0.6, 0.03])


freq_slider = Slider(ax, 'Freq', 0.1, 5.0, valinit=1)
freq_slider.on_changed(sliders_on_changed)


ax = fig.add_axes([0.025, 0.5, 0.15, 0.15])
color_radios = RadioButtons(ax, ('red', 'blue', 'green'), active=1)

def color_on_clicked(label):
    line.set_color(label)
    fig.canvas.draw_idle()

color_radios.on_clicked(color_on_clicked)

plt.show()






