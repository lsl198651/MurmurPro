import shutil
import json
import os
import torch
import torch.nn as nn
import os
import shutil
import librosa
import librosa.display
import numpy as np
from util.helper_code import *
from pyts.image import GramianAngularField, MarkovTransitionField
import PIL.Image as pil_image
from scipy.signal import butter, lfilter
import pywt


def getMelFeaturesAndFreq(recording_features, targetFreq=4000):
    Mel_Spectrum = Mel_Time_Frequency_Spectrum(recording_features, targetFreq)
    recording_features = bandpass_filter(
        recording_features, 10, 400, targetFreq, 6)
    Mel_Spectrum2 = Mel_Time_Frequency_Spectrum_2(
        recording_features, targetFreq)
    Mel_Spectrum = np.concatenate(
        [Mel_Spectrum[:, :-1], Mel_Spectrum2], axis=0)

    return Mel_Spectrum,


################################################################################
# mel time_fre spectrum
################################################################################
def Mel_Time_Frequency_Spectrum(signal, Fs=4000):

    EPS = 1E-6
    melspectrogram = librosa.feature.melspectrogram(y=signal, sr=Fs, n_mels=16,
                                                    hop_length=8, win_length=20, n_fft=256)
    lms = np.log(melspectrogram + EPS)
    return lms


def myDownSample_2d(data, sample_Fs, targetFreq):
    step = sample_Fs // targetFreq
    newData = np.array([data[:, i] for i in range(
        data.shape[1]) if i % step == 0]).transpose([1, 0])
    return newData


def Mel_Time_Frequency_Spectrum_2(signal, Fs):
    EPS = 1E-6
    coef, freqs = pywt.cwt(signal, np.arange(1, 17), 'cgau3')
    coef = np.abs(coef)
    coef = myDownSample_2d(coef, Fs, Fs/8)
    lms = np.log(coef + EPS)
    return lms


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    # print(low,high)
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def get_mfcc(wavform):
    """提取wav的mfcc特征

    Args:
        wavform (float): 心音的时域信号
    """
    # 读取音频文件
    # y, sr = librosa.load(wav_file, sr=None)
    # 提取mfcc特征
    mfccs = librosa.feature.mfcc(wavform, sr=4000, n_mfcc=40)
    # return mfccs
    return mfccs


def get_logmel_feature(wavform, fs=4000):
    """提取logmel特征"""
    # 读取音频文件
    # y, sr = librosa.load(wavform, sr=4000)
    # 提取特征
    mel = librosa.feature.melspectrogram(
        y=wavform, sr=fs, n_mels=128, n_fft=256)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel


def get_MarkovTransitionField(wavform):
    """求马尔科夫转移场特征"""
    X = np.array([wavform])
    mtf = MarkovTransitionField(n_bins=8)
    X_mtf = mtf.fit_transform(X)
    return X_mtf[0]


def get_GramianAngularField(wavform):
    """求格拉姆角场特征

    Args:
        wavform (_type_): 心音始于信号
    """

    X = np.array(wavform).reshape(-1, 1)
    # Compute Gramian angular fields
    gasf = GramianAngularField(method='summation')
    X_gasf = gasf.fit_transform(X)
    img = pil_image.fromarray(X_gasf[0])
    image = img.resize((150, 150), resample=pil_image.BICUBIC)
    image = np.array(image)
    return image


def get_sample_entropy(wavform):
    """求样本熵特征

    Args:
        wavform (float32): 心音时域信号
    """
    # 读取音频文件
    # y, sr = librosa.load(wav_file, sr=None)
    # 提取样本熵特征
    sample_spre = entropy.sample_entropy(wavform)
    # return sample_spre
    return sample_spre


def get_distribution_entropy(wavform):
    """求分布熵特征

    Args:
        wavform (float): 时域信号
    """
    # 读取音频文件
    # y, sr = librosa.load(wav_file, sr=None)
    # 提取分布熵特征


def get_fuzzy_entropy(wavform):
    """求模糊熵特征

    Args:
        wavform (float): _description_
    """
    # 读取音频文件
    # y, sr = librosa.load(wav_file, sr=None)
    # 提取模糊熵特征
    mohu_spre = entropy.fuzzy_entropy(wavform)
    # return mohu_spre
    return mohu_spre


def get_time_feature(wavform):
    """求信号的时域特征

    Args:
        wavform (_type_): _description_
    """


def get_frequency_feature(wavform):
    """求信号的频域特征

    Args:
        wavform (_type_): _description_
    """


def get_features_mod(data):
    # Extract the age group, sex and the pregnancy status features
    age_group = get_age(data)
    age_list = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    is_pregnant = get_pregnancy_status(data)
    if age_group not in ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']:
        if is_pregnant:
            age = 'Young Adult'
        else:
            age = 'Child'
    else:
        age = age_group
    age_fea = str(age_list.index(age))
    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)
    if compare_strings(sex, 'Female'):
        sex_features = '0'
    elif compare_strings(sex, 'Male'):
        sex_features = '1'
    if is_pregnant:
        preg_fea = '1'
    else:
        preg_fea = '0'
    return age_fea + sex_features + preg_fea


class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            params = json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage():
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, loss):
        self.total += loss
        self.steps += 1

    def __call__(self):
        return (self.total/float(self.steps))


def save_checkpoint(state, modelname, split, checkpoint):
    filename = os.path.join(checkpoint, 'last{}.pth.tar'.format(split))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint)
    torch.save(state, filename)
    shutil.copyfile(filename, os.path.join(
        checkpoint, f"{modelname}_model_best_{split}.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint):
        raise ("File Not Found Error {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model


def initialize_weights(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        nn.init.ones_(m.weight.data)
