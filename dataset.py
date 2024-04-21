# -*- coding: UTF-8 -*- #
"""
@filename:dataset.py
@author:Young
@time:2024-04-20
"""
import os
import json
import scipy
import librosa
import argparse
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

"""
定义绘图方法
save_or_show：保存或显示绘图
plot_fft：绘制FFT幅度图
plot_waveform：绘制波形图
plot_mfccs：绘制MFCCs图
plot_spectrogram：绘制频谱图
"""

def save_or_show(save, filename):
    """可先show图像，再save绘制的图像"""
    if save:
        # TODO: Add a check here because the filename should not be None
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False) # forward-尺寸缩放
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()

def plot_fft(signal, sample_rate, save=False, f=None):
    """Plot the amplitude of the FFT of a signal."""
    yf = scipy.fft.fft(signal)
    period = 1/sample_rate
    samples = len(yf)
    xf = np.linspace(0.0, 1/(2.0 * period), len(signal)//2)
    plt.plot(xf / 1000, 2.0 / samples * np.abs(yf[:samples//2]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title("FFT")
    save_or_show(save, f)

def plot_waveform(signal, sample_rate, save=False, f=None):
    """Plot waveform in the time domain."""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    save_or_show(save, f)

def plot_mfccs(mfccs, save=False, f=None):
    """Plot the mfccs spectrogram."""
    dims = mfccs.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims + 1)])
    librosa.display.specshow(mfccs, x_coords=x_coords, x_axis='time',
                             hop_length=512)
    plt.colorbar()
    plt.xlabel("Time (seconds)")
    plt.title("MFCCs")
    plt.tight_layout()
    save_or_show(save, f)

def plot_spectrogram(spec, save=False, f=None):
    """Plot spectrogram's amplitude in DB"""
    fig, ax = plt.subplots()
    dims = spec.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims + 1)])
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                                   x_coords=x_coords, y_axis='log',
                                   x_axis='time', ax=ax)

    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

"""
从音乐数据集中提取特征，并保存到JSON文件中
preprocess_dataset_mfcc：从音乐数据集中提取MFCC，并保存到JSON文件中
preprocess_dataset_spectro：从音乐数据集中提取频谱图，并保存到JSON文件中
preprocess_dataset：根据命令行参数选择提取MFCC或频谱图，并调用相应的处理函数
"""

def preprocess_dataset_mfcc(dataset_path, json_path, n_mfcc, n_fft,
                            hop_length, samples_to_consider):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to data
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of
                        samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    i = 0
    # loop through all sub-dirs
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        # We did not use enumerate in the loop because the index will be
        # increased even in the case that a directory was skipped.
        if "_background_noise_" in dirpath:
            continue

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency
                # among different files
                signal, sample_rate = librosa.load(file_path, sr=None)

                # drop audio files with less than pre-decided number of samples
                # TODO: Maybe pad all these signals with zeros in the end
                if len(signal) >= samples_to_consider:

                    # ensure consistency of the length of the signal
                    signal = signal[:samples_to_consider]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate,
                                                 n_mfcc=n_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i))

            # Increase the counter
            i += 1

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def preprocess_dataset_spectro(dataset_path, json_path, samples_to_consider,
                               n_fft=256, hop_length=512):
    """
    Create a json with the spectrograms.

    Ideas taken from
    https://www.tensorflow.org/tutorials/audio/simple_audio
    TODO: Remove duplicate code.
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "spectro": [],
        "files": []
    }

    i = 0
    # loop through all sub-dirs
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        # We did not use enumerate in the loop because the index will be
        # increased even in the case that a directory was skipped.
        if "_background_noise_" in dirpath:
            continue

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency
                # among different files
                signal, sample_rate = librosa.load(file_path, sr=None)

                # drop audio files with less than pre-decided number of samples
                # TODO: Maybe pad all these signals with zeros in the end
                if len(signal) >= samples_to_consider:

                    # ensure consistency of the length of the signal
                    signal = signal[:samples_to_consider]

                    # extract spectrogram
                    spectrogram = librosa.stft(signal[:samples_to_consider],
                                               n_fft=n_fft,
                                               hop_length=hop_length)
                    spectrogram = np.abs(spectrogram)

                    # store data for analysed track
                    data["spectro"].append(spectrogram.T.tolist())
                    data["labels"].append(i)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i))

            # Increase the counter
            i += 1

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def preprocess_dataset(dataset_path, json_path, n_mfcc, n_fft, l_hop, samples):
    """Choose between the two features."""
    if "mfcc" in json_path:
        preprocess_dataset_mfcc(dataset_path, json_path, n_mfcc, n_fft, l_hop,
                                samples)
    else:
        preprocess_dataset_spectro(dataset_path, json_path, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose calculated features")
    parser.add_argument("features", choices=["mfccs", "spectrogram"], type=str,
                        help="Choose calculated features") # 特征提取方式
    parser.add_argument("path", type=str, help="Give the dataset's path") # 数据集地址
    parser.add_argument("samples", type=int, help="Samples to consider"
                        "according to the signal's sampling rate") # 信号采样率
    parser.add_argument("n_mfcc", type=int, help="Number of mel-bands",
                        default=40, nargs='?')
    parser.add_argument("n_fft", type=int, help="FFT's window size for the "
                        "mel-spectrogram", default=1103, nargs='?')
    parser.add_argument("l_hop", type=int, help="Number of samples between "
                        "successive frames", default=160, nargs='?')
    # Read arguments
    args = parser.parse_args()

    # Check if given directory exists.
    if not os.path.isdir(args.path):
        print("Given directory does not exist")
        exit(1)

    if args.features == "mfccs":
        json_path = (f"mfcc_{args.samples}_{args.n_mfcc}_{args.n_fft}_"
                     f"{args.l_hop}_{args.path}.json")
    else:
        json_path = "data_spectro.json"

    preprocess_dataset(args.path, json_path, args.n_mfcc, args.n_fft,
                       args.l_hop, args.samples)