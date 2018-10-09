import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def three_temp_model(x, a, b, c, d, e, f):
    return a * np.exp(b * x) + c * np.exp(d * x) + e * np.exp(f * x)

def apply_model(data, model, p0):
    return curve_fit(model, data[0], data[1], p0=p0)

def low_pass_filter(fft_signal, cut_off_freq):
    for i, item in enumerate(fft_signal[0]):
        if fft_signal[0][i] > cut_off_freq:
            fft_signal[0][i] = 0
            fft_signal[1][i] = 0 

def build_data_from_model(params, scope):
    ax = np.linspace(scope[0]/1000, (scope[1]-scope[0])/1000, (scope[1]-scope[0]))
    return (ax, three_temp_model(ax, *params))

def load_file(path):
    ax, data = np.genfromtxt(path, unpack=True)
    return (ax, -1*data)

def sub(x, y):
    return (x[0], x[1] - y[1])


def plot_data(plot_data, out_name, **kwargs):
    for key in kwargs.keys():
        if key == "scope":
            plt.xlim(kwargs[key][0], kwargs[key][1])

    plt.plot(plot_data[0], plot_data[1], "bx", label="data")

    for key in kwargs.keys():
        if key == "model_data":
            plt.plot(kwargs[key][0], kwargs[key][1], "r-", label="model")


    #plt.show()
    print("savefig: {}".format(out_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "img", "eval", out_name))
    plt.clf()

def set_time_offset(data, offset):
    return (data[0][offset:], data[1][offset:])

def main():
    data_paths = []
    blank_probe_path = ""
    blank_probe_filename = "002_a000_b0_e245_FILM.txt"

    for root, dirs, files in os.walk(".."):
        if root == "../data":
            for item in files:
                if item  == blank_probe_filename:
                    blank_probe_path = os.path.join(root, item)
                    continue

                if item.split(".")[1] == "txt":
                    data_paths.append(os.path.join(root, item))

    # plot blank probe
    data = load_file(blank_probe_path)
    data = set_time_offset(data, 350)

    # apply three temp model
    p0 = []

    # predict p0
    p0.append(data[1][1])
    p0.append(-1)
    p0.append(data[1][1]/10)
    p0.append(-1)
    p0.append(data[1][1]/100)
    p0.append(-1)


    (params, cov) = apply_model(data, three_temp_model, p0=p0)
    model_data = build_data_from_model(params, scope=(350, 10000))
    
    # plot
    plot_data(data, blank_probe_path.split("/")[-1].replace("txt", "pdf"), model_data=model_data)

    # plot all
    for path in data_paths:
        data = load_file(path)
        data = set_time_offset(data, 350)
        plot_data(data, path.split("/")[-1].replace("txt", "pdf"), scope=(2, 3))

        # correct data with three temp modell
        signal = sub(data, model_data)
        plot_data(signal, path.split("/")[-1].replace("txt", "corr.pdf"), scope=(2, 3))

        # apply fft on signal
        fft_freq = np.fft.fftfreq(data[0].size, d=1./1000)
        fft_val = np.fft.fft(signal[1])
        fft_signal_real = (fft_freq, abs(fft_val.real))
        plot_data(fft_signal_real, path.split("/")[-1].replace("txt", "fft.pdf"), scope=(11.5, 13.5))

        # apply low pass filter at 16 Ghz
        low_pass_filter(fft_signal_real, 16)
        signal = np.fft.ifft(fft_signal_real[1])
        ax = np.fft.fftfreq(signal.size, d=1./1000)

        data = (ax, signal.real)
        plot_data(data, path.split("/")[-1].replace("txt", "lowpass.pdf"), scope=(200, 300))

if __name__ == "__main__":
    main()