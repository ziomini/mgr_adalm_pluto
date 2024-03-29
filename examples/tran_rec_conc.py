import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 50e6 # MHz
center_freq = 24e8 # Hz 2.4GHz
num_samps = 10000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps*10
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Create transmit waveform (QPSK, 16 samples per symbol)
# num_symbols = 1000
# x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
# x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
# x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
# x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
# samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
N = 10000
t = np.arange(N)/sample_rate
f = np.linspace(start=1e3, stop=(2.5e5), num=N)
# f = np.arange(start=1e3, stop=(25e2+ 1e3), step=25)
print(len(t))
print(len(f))
tx_samples = 0.5*np.exp(2.0j*np.pi*np.multiply(f,t)) # Simulate chirp 
tx_samples = np.concatenate((tx_samples, np.zeros(N)))
print(np.min(np.real(tx_samples)), np.max(np.real(tx_samples)))
print(np.min(np.imag(tx_samples)), np.max(np.imag(tx_samples)))

psd = np.abs(np.fft.fftshift(np.fft.fft(tx_samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))
# Plot freq domain
plt.figure(3)
plt.plot(f/2e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")

tx_samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(tx_samples) # start transmitting

# Clear buffer just to be safe
for i in range (0, 1):
    raw_data = sdr.rx()

# Receive samples
total_samples = []
fmcw_samples = []
fmcw_current_sample = []
total_samples = np.asarray(total_samples)
for i in range(0,1):
    rx_samples = sdr.rx()
    total_samples = np.concatenate((total_samples, rx_samples))
    # fmcw_current_sample = np.multiply(tx_samples, rx_samples)
    fmcw_current_sample = rx_samples
    fmcw_samples = np.concatenate((fmcw_samples, fmcw_current_sample))
    # total_samples = rx_samples
# print(total_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# Calculate power spectral density (frequency domain version of signal)
psd = np.abs(np.fft.fftshift(np.fft.fft(total_samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))

# Plot time domain

# Plot freq domain
plt.figure(1)
plt.plot(f/2e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")

plt.figure(2)
NFFT = 1024
plt.specgram(fmcw_samples, NFFT=NFFT, Fs=sample_rate, noverlap=900)
plt.show()

fmcw_sig = tx_samples
for i in range(0,4):
    fmcw_sig = np.concatenate((fmcw_sig, tx_samples))

plt.figure(3)
sig_fmcw = np.multiply(fmcw_sig, fmcw_samples) 
spec = np.abs(np.fft.fftshift(np.fft.fft(sig_fmcw)))
plt.plot(f/2e6, 10*np.log10(spec))
plt.title("After multi")
plt.xlabel("Freq [MHz]")