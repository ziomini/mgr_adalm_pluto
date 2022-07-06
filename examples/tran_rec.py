import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 2e6 # Hz
center_freq = 24e8 # Hz 2.4GHz
num_samps = 100000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
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
f = np.arange(start=1e3, stop=(25e4+ 1e3), step=25)
print(len(t))
print(len(f))
samples = 0.5*np.exp(2.0j*np.pi*np.multiply(f,t)) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
print(np.min(np.real(samples)), np.max(np.real(samples)))
print(np.min(np.imag(samples)), np.max(np.imag(samples)))

psd = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))
# Plot freq domain
plt.figure(3)
plt.plot(f/2e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")

samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Receive samples
total_samples = []
total_samples = np.asarray(total_samples)
for i in range(0,10):
    rx_samples = sdr.rx()
    # total_samples = np.concatenate(total_samples, rx_samples, axis=None)
    total_samples = rx_samples
# print(total_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# Calculate power spectral density (frequency domain version of signal)
psd = np.abs(np.fft.fftshift(np.fft.fft(total_samples)))**2
psd_dB = 10*np.log10(psd)
f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))

# Plot time domain
plt.figure(0)
plt.plot(np.real(total_samples))
plt.plot(np.imag(total_samples))
plt.xlabel("Time")

# Plot freq domain
plt.figure(1)
plt.plot(f/2e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")
plt.show()