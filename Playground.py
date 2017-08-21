import scipy.signal as sps
import matplotlib.pyplot as plt
import numpy as np
import PulseGeneration as pg

sampling_rate = 20000
duration = 2
hz = 5

pulse, t = pg.extended_square_pulse(sampling_rate, duration, hz, 0.5)
pulse2, t2 = pg.square_pulse(sampling_rate, duration, hz, 0.5)

plt.figure()
# plt.hold(True)
plt.plot(t, pulse)
plt.plot(t2, pulse2)
plt.show()