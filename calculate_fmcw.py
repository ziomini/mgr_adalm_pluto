from termios import FF1
import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

def nlfm_hamming(time_vector, bandwidth, center_freq, Niter = 4):
    time = np.copy(time_vector)
    time_diff = np.max(time) - np.min(time)
    time = np.subtract(time, time_diff)
    dt = time[2] - time[1]
    Nt = (np.arange(len(time))+1)/len(time)

    gg  = 0.54 - 0.46 * np.cos(2.0*np.pi*Nt)
    g0 = 1/gg
    up_factor = int(np.round(len(g0)/len(time)))
    print(up_factor)
    c = np.cumsum(g0[1::up_factor])
    for i in range(1, Niter):
        gg  = 0.54 - 0.46 * np.cos(2.0*np.pi*c / c[-1])
        g1 = 1 / gg
        c = np.cumsum(g1)

    c = (2 * c / c[-1] -1)
    phiprim = np.pi * bandwidth * c + 2 * np.pi * center_freq
    phi = np.cumsum(phiprim * dt)
    return np.exp(1.0j * phi)


def calculate_spec(samples, db_scale=True, window = 'hamming'):
    if window == 'hamming':
        hamming_samples = np.hamming(len(samples))
        samples = np.multiply(samples, hamming_samples)

    fft_result = np.fft.fftshift(np.fft.fft(samples))

    spec = np.abs(fft_result)
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
f2 = 50e6
f1_norm = (f1)/sample_rate
f2_norm = (f2)/sample_rate
dfdt = (f2-f1)/(num_samps * (1/sample_rate))


singal_real = False
signal_modified = True
save_variables = False
plot_fmcw_spec = True
save_result_to_output_file = True
secondary_diff = True
output_filename = "out.txt"
window_len = 100


##################
# FMCW functions #
##################


def plot_norm_spec(samples_spec_in, fig_num=1, title=None):
    samples_spec = np.copy(samples_spec_in)
    peaks = find_peaks(samples_spec)[0]
    index = None
    index = np.argmax(samples_spec)
    
    samples_spec = samples_spec[(index-window_len):(index+window_len)]



    x_axis = np.linspace(0, len(samples_spec),len(samples_spec))
    plt.figure(fig_num)
    plt.plot(x_axis, samples_spec)
    if title is not None:
        plt.title(title)
    peaks = find_peaks(samples_spec)[0]
    peaks_values = samples_spec[peaks]
    ind = np.argpartition(peaks_values, -2)[-2:]
    y_top = peaks_values[ind]
    ind = peaks[ind]
    for i, j in zip(x_axis[ind], y_top):
        print("%s and %s" % (i, j))
        plt.text(i, j, '({}, {})'.format(i, j))
    ind = np.sort(ind)
    plt.text(0, 5, 'diff %s' % (x_axis[ind[1]] - x_axis[ind[0]]))
    # plt.text(0, 5.2, 'diff %s' % (x_axis[ind[2]] - x_axis[ind[0]]))
    diff = (x_axis[ind[1]] - x_axis[ind[0]])
    diff = (diff*sample_rate)/N
    diff_dist = (c * abs(diff))/(2*dfdt)
    plt.text(0, 5.4, 'diff dist %s' % (diff_dist))
    # plt.show()
    exp_name = 'none'
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    plt.savefig('exp_d_%s' % (exp_name))


def calculate_fmcw(samples_spec_in):
    samples_spec = np.copy(samples_spec_in)
    peaks = find_peaks(samples_spec)[0]
    index = None
    index = np.argmax(samples_spec)
    
    samples_spec = samples_spec[(index-window_len):(index+window_len)]

    x_axis = np.linspace(0, len(samples_spec),len(samples_spec))


    peaks = find_peaks(samples_spec)[0]

    peaks_values = samples_spec[peaks]
    peak_indexes = np.argpartition(peaks_values, -2)[-2:]  # pick top 2 peaks
    peak_indexes = peaks[peak_indexes]

    # for i, j in zip(x_axis[peak_indexes], peaks_values[peak_indexes]):
    #     if save_variables:
    #         print("%s and %s" % (i, j))
    peak_indexes = np.sort(peak_indexes)
    diff = (x_axis[peak_indexes[1]] - x_axis[peak_indexes[0]])
    diff = (diff*sample_rate)/N
    diff_dist = (c * abs(diff))/(2*dfdt)
    if save_variables:
        print('%s * %s / 2 * %s' % (c, diff, dfdt))
    print('dist: %s' % diff_dist)

    if save_result_to_output_file:
        file = open(output_filename, 'a')
        exp_name = 'none'
        if len(sys.argv) > 1:
            exp_name = sys.argv[1]
        file.write("real distance: %s\tradar distance\t: %s\n" % (exp_name, diff_dist))

    if secondary_diff:
        x_axis = np.linspace(-0.5, 0.5, len(samples_spec))
        peaks = find_peaks(samples_spec)[0]
        peaks_values = samples_spec[peaks]
        peak_indexes = np.argpartition(peaks_values, -3)[-3:]  # pick top 6 peaks
        peak_indexes = peaks[peak_indexes]

        # for i, j in zip(x_axis[peak_indexes], peaks_values[peak_indexes]):
        #     if save_variables:
        #         print("%s and %s" % (i, j))
        peak_indexes = np.sort(peak_indexes)
        diff = (x_axis[peak_indexes[2]] - x_axis[peak_indexes[1]])
        diff = (diff*sample_rate)/N
        diff_dist = (c * abs(diff))/(2*dfdt)
        print('secondary dist: %s' % diff_dist)
        


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

### Generate signal
if signal_modified:
    if singal_real:
        tx_samples = 0.5*np.sin(2.0*np.pi*np.multiply(f, t))  # Simulate real chirp
    else:
        tx_samples = 0.5*np.exp(2.0j*np.pi*np.multiply(f, t)
                                )  # Simulate complex chirp
else:
    tx_samples = nlfm_hamming(t, sample_rate, 0) #TODO

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


# fmcw_samples = np.multiply(fmcw_samples, np.max(total_samples))

fmcw_result_samples = np.multiply(total_samples, fmcw_samples)

if plot_fmcw_spec:
    plot_norm_spec(calculate_spec(fmcw_result_samples), 2, "FMCW Result")

calculate_fmcw(calculate_spec(fmcw_result_samples))
