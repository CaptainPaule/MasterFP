import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def three_temp_model(x, a, b, c, d, e, f):
    return a * np.exp(b * x) + c * np.exp(d * x) + e * np.exp(f * x)

def lorentz_model(w, w0, gamma):
    return 1. / ((w**2 - w0**2)**2 + gamma**2 * w0**2)

def linear_model(x, a, b):
    return a*x + b

def apply_model(data, model, p0):
    return curve_fit(model, data[0], data[1], p0=p0)

def peak_detect(data):
    peak = 0
    for item in data:
        if abs(item) > peak:
            peak = abs(item)

    return peak

def low_pass_filter(fft_signal, cut_off_freq):
    for i, item in enumerate(fft_signal[0]):
        if fft_signal[0][i] > cut_off_freq:
            fft_signal[0][i] = 0
            fft_signal[1][i] = 0 

def build_data_from_model(params, scope, model):
    ax = np.linspace(scope[0]/1000, (scope[1]-scope[0])/1000, 9650)
    return (ax, model(ax, *params))

def load_file(path):
    ax, data = np.genfromtxt(path, unpack=True)
    return (ax, -1*data)

def sub(x, y):
    return (x[0], x[1] - y[1])


def plot_data(plot_data, out_name, **kwargs):
    for key in kwargs.keys():
        if key == "xscope":
            plt.xlim(kwargs[key][0], kwargs[key][1])
        if key == "yscope":
            plt.ylim(kwargs[key][0], kwargs[key][1])


    plt.plot(plot_data[0], plot_data[1], "bx", label="data")
    for key in kwargs.keys():
        if key == "model_data":
            plt.plot(kwargs[key][0], kwargs[key][1], "r-", label="model")


    #plt.show()
    plt.xlabel(r"$time$ / ns")
    plt.ylabel(r"r")
    plt.xticks([200, 220, 240, 260, 280, 300], [2, 2.2, 2.4, 2.6, 2.8, 3.0])
    plt.legend()
    print("savefig: {}".format(out_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "img", "eval", out_name))
    plt.clf()

def set_time_offset(data, offset):
    return (data[0][offset:], data[1][offset:])


def main():
    p0_store = {
                "005_a000_b0_e245_G4.txt"    : [12.5, 0.1],
                "006_a000_b0_e245_G3.txt"    : [12.5, 0.1],
                "007_a000_b0_e245_G2.txt"    : [12.5, 0.1],
                "008_a000_b0_e245_G1.txt"    : [12.5, 0.1],
                "009_a000_b0_e245_G4_90.txt" : [12.5, 0.1],
                "011_a000_b0_e245_G4_45.txt" : [12.5, 0.1],
                }

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
    model_data = build_data_from_model(params, scope=(350, 10000), model=three_temp_model)
    errors = np.sqrt(np.diag(cov))

    print("a = {0} +- {1}".format(params[0], errors[0]))
    print("b = {0} +- {1}".format(params[1], errors[1]))
    print("c = {0} +- {1}".format(params[2], errors[2]))
    print("d = {0} +- {1}".format(params[3], errors[3]))
    print("e = {0} +- {1}".format(params[4], errors[4]))
    print("f = {0} +- {1}".format(params[5], errors[5]))
    
    # plot
    plot_data(data, blank_probe_path.split("/")[-1].replace("txt", "pdf"), model_data=model_data)

    # plot all
    freq_peaks = []

    for path in data_paths:
        data = load_file(path)
        data = set_time_offset(data, 350)
        plot_data(data, path.split("/")[-1].replace("txt", "pdf"), xscope=(2, 3))

        # correct data with three temp modell
        signal = sub(data, model_data)
        plot_data(signal, path.split("/")[-1].replace("txt", "corr.pdf"), xscope=(2, 3), yscope=(-0.0005, 0.0005))

        # apply fft on signal
        fft_freq = np.fft.fftfreq(signal[0].size, d=1./1000)
        fft_val = np.fft.fft(signal[1], signal[1].size)
        fft_signal_tmp = (fft_freq, fft_val.real)
        fft_signal_real = (fft_freq, abs(fft_val.real))

        # fit lorentz
        # select p0
        #p0 = p0_store[path.split("/")[-1]]

        #(params, cov) = apply_model(fft_signal_real, lorentz_model, p0=p0)
        #model_data_fft = build_data_from_model(params, scope=(0, 20000), model=lorentz_model)
        #plot_data(fft_signal_real, path.split("/")[-1].replace("txt", "fft.pdf"), xscope=(7.5, 15.5), yscope=(0.0, 0.25), model_data=model_data_fft)
        #freq_peaks.append(peak_detect(model_data_fft[1]))

        #errors = np.sqrt(np.diag(cov))
        #print("omega = {0} +- {1}".format(params[0], errors[0]))
        #print("gamma = {0} +- {1}".format(params[1], errors[1]))

        # apply low pass filter at 16 Ghz
        low_pass_filter(fft_signal_tmp, 16)
        signal = np.fft.ifft(fft_signal_tmp[1], fft_signal_tmp[1].size)
        ax = np.fft.fftfreq(signal.size, d=1./1000)

        data = (ax, signal.real)
        plot_data(data, path.split("/")[-1].replace("txt", "lowpass.pdf"), xscope=(200, 300), yscope=(-0.0005, 0.0005))

    print(freq_peaks)
    # print amplitudes related to grid depth
    #ax = [23, 17, 14, 7]
    #(params, cov) = apply_model((ax, freq_peaks), linear_model, p0=[0, 0])
    #model_data = build_data_from_model(params, scope=(7000, 30000), model=linear_model)
    #plot_data((ax, freq_peaks), "amplitudes.pdf", model_data=model_data)
    #errors = np.sqrt(np.diag(cov))
    #print("a = {0} +- {1}".format(params[0], errors[0]))
    #print("b = {0} +- {1}".format(params[1], errors[1]))

if __name__ == "__main__":
    main()