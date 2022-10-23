from termios import FF1
import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys


def calculate_spec(samples, db_scale=True):
    spec = np.abs(np.fft.fftshift(np.fft.fft(samples)))
    if db_scale:
        return np.log10(spec)
    else:
        return spec


#######################
# Variables to change #
#######################

c = 299792458

sample_rate = 50e6  # MHz
center_freq = 2.40e9  # Hz 2.45GHz
num_samps = 10000000  # number of samples per call to rx()
f1 = 1e3
f2 = 25e6
f1_norm = (f1)/sample_rate
f2_norm = (f2)/sample_rate
dfdt = (f2-f1)/(num_samps * (1/sample_rate))


singal_real = True
save_variables = False
plot_fmcw_spec = False
save_result_to_output_file = True
output_filename = "out.txt"


##################
# FMCW functions #
##################


def plot_norm_spec(samples_spec, fig_num=1, title=None):
    x_axis = np.linspace(-0.5, 0.5, len(samples_spec))
    plt.figure(fig_num)
    plt.plot(x_axis, samples_spec)
    if title is not None:
        plt.title(title)
    peaks = find_peaks(samples_spec)[0]
    peaks_values = samples_spec[peaks]
    ind = np.argpartition(peaks_values, -4)[-4:]
    y_top = peaks_values[ind]
    ind = peaks[ind]
    for i, j in zip(x_axis[ind], y_top):
        print("%s and %s" % (i, j))
        plt.text(i, j, '({}, {})'.format(i, j))
    ind = np.sort(ind)
    plt.text(0, 5, 'diff %s' % (x_axis[ind[1]] - x_axis[ind[0]]))
    plt.text(0, 5.2, 'diff %s' % (x_axis[ind[2]] - x_axis[ind[3]]))
    diff = (x_axis[ind[1]] - x_axis[ind[0]])
    diff = diff*50e6
    diff_dist = (c * abs(diff))/(2*dfdt)
    plt.text(0, 5.4, 'diff dist %s' % (diff_dist))
    plt.show()


def calculate_fmcw(samples_spec):
    x_axis = np.linspace(-0.5, 0.5, len(samples_spec))
    peaks = find_peaks(samples_spec)[0]
    peaks_values = samples_spec[peaks]
    peak_indexes = np.argpartition(peaks_values, -4)[-4:]  # pick top 4 peaks
    peak_indexes = peaks[peak_indexes]

    # for i, j in zip(x_axis[peak_indexes], peaks_values[peak_indexes]):
    #     if save_variables:
    #         print("%s and %s" % (i, j))
    peak_indexes = np.sort(peak_indexes)
    diff = (x_axis[peak_indexes[1]] - x_axis[peak_indexes[0]])
    diff = diff*sample_rate
    diff_dist = (c * abs(diff))/(2*dfdt)
    if save_variables:
        print('%s * %s / 2 * %s' % (c, diff, dfdt))
    print('dist: %s' % diff_dist)

    if save_result_to_output_file:
        file = open(output_filename, 'a')
        exp_name = 'none'
        if len(sys.argv) > 1:
            exp_name = sys.argv[1]
        file.write("real distance: %s\tradar distance\n: %s" % (exp_name, diff_dist))


#########
# Start #
#########
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
# filter cutoff, just set it to the same as sample rate
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(center_freq)
# Increase to increase tx power, valid range is -90 to 0 dB
sdr.tx_hardwaregain_chan0 = -10

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
# dB, increase to increase the receive gain, but be careful not to saturate the ADC
sdr.rx_hardwaregain_chan0 = 0.0


N = int(num_samps)
print("N: %s" % N)
t = np.arange(N)/sample_rate
f = np.linspace(start=f1, stop=(f2), num=N)
print(f)
# f = np.arange(start=1e3, stop=(25e2+ 1e3), step=25)
print(len(t))
print(len(f))
if singal_real:
    tx_samples = 0.5*np.sin(2.0*np.pi*np.multiply(f, t))  # Simulate real chirp
else:
    tx_samples = 0.5*np.exp(2.0j*np.pi*np.multiply(f, t)
                            )  # Simulate complex chirp


# tx_samples = np.concatenate((tx_samples, np.zeros(N)))
print(np.min(np.real(tx_samples)), np.max(np.real(tx_samples)))
print(np.min(np.imag(tx_samples)), np.max(np.imag(tx_samples)))

spec_tx = calculate_spec(tx_samples)
f = np.linspace(sample_rate/-2, sample_rate/2, len(spec_tx))

fmcw_samples = []
for i in range(0, 1):
    fmcw_samples = np.concatenate((fmcw_samples, tx_samples))

# The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
tx_samples *= 2**14
# Start the transmitter
sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
sdr.tx(tx_samples)  # start transmitting


# Clear buffer just to be safe
for i in range(0, 5):
    raw_data = sdr.rx()


# Receive samples
total_samples = []
total_samples = np.asarray(total_samples)
for i in range(0, 1):
    rx_samples = sdr.rx()
    total_samples = np.concatenate((total_samples, rx_samples))

# Stop transmitting
sdr.tx_destroy_buffer()


fmcw_samples = np.multiply(fmcw_samples, np.max(total_samples))

fmcw_result_samples = np.multiply(total_samples, fmcw_samples)

if plot_fmcw_spec:
    plot_norm_spec(calculate_spec(fmcw_result_samples), 2, "FMCW Result")

calculate_fmcw(calculate_spec(fmcw_samples))
