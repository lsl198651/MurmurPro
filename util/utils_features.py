import json
import shutil

import librosa
import librosa.display
import torch
import torch.nn as nn

from util.helper_code import *


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
    mel = librosa.feature.melspectrogram(y=wavform, sr=fs, n_mels=128, n_fft=256)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel


def get_lami_angle(wavform):
    """求格拉姆角场特征

    Args:
        wavform (_type_): 心音始于信号
    """


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
    distrub_spre = entropy.distribution_entropy(wavform)
    # return distrub_spre
    return distrub_spre


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


def get_gelamr_abgle(wavform):
    """求格拉莫角域特征

    Args:
        wavform (_type_): _description_
    """
    # 读取音频文件
    # y, sr = librosa.load(wav_file, sr=None)
    # 提取格拉莫角域特征
    gelamr_abgle = entropy.gelamr_abgle(wavform)
    # return gelamr_abgle
    return gelamr_abgle


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
        return (self.total / float(self.steps))


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
