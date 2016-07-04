import scipy.signal as sps
import matplotlib.pyplot as plt
import numpy as np

sampling_rate = 20000
duration = 2
hz = 5

t = np.linspace(0, duration, duration*sampling_rate, endpoint=False)
signal = sps.square(2 * np.pi * hz * t, duty = 0.1)
plt.plot(t, signal)
plt.ylim(-2, 2)
plt.show()