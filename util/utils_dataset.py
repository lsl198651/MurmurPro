import os
import shutil
import random
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile
import csv
import numpy as np
import pandas as pd
# from BEATs_def import mkdir
# import soundfile as sf
# from BEATs_def import get_wav_data
import pandas as pd
from util.helper_code import *
from util.utils_features import get_features_mod, get_logmel_feature


def mkdir(path):
    folder = os.path.exists(path)
    # judge wether make dir or not
    if not folder:
        os.makedirs(path)


def csv_reader_cl(file_name, clo_num):
    """read csv file by column
    """
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


#
def csv_reader_row(file_name, row_num):
    """read the csv row_num-th row"""
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        row = list(reader)
    return row[row_num]


def copy_file(src_path, folder_path, patient_id_list, mur, position):
    """将所有文件复制到目标目录"""
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)

            txtname = src_path + "\\" + patient_id + ".txt"
            wavname = src_path + "\\" + patient_id + pos + ".wav"
            heaname = src_path + "\\" + patient_id + pos + ".hea"
            tsvname = src_path + "\\" + patient_id + pos + ".tsv"

            if os.path.exists(txtname):
                shutil.copy(txtname, target_dir + "\\")
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir + "\\")
            if os.path.exists(heaname):
                shutil.copy(heaname, target_dir + "\\")
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir + "\\")


def copy_wav_file(src_path, folder_path, patient_id_list, mur, position):
    """将wav和tsv文件复制到目标目录"""
    count = 0
    # 1. make dir
    mur_dir = folder_path + "\\" + mur
    if not os.path.exists(mur_dir):
        os.makedirs(mur_dir)
    # 2. copy file
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)
            wavname = src_path + "\\" + patient_id + pos + ".wav"
            tsvname = src_path + "\\" + patient_id + pos + ".tsv"
            txtname = src_path + "\\" + patient_id + ".txt"
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir + "\\")
                count += 1
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir + "\\")
            if os.path.exists(txtname):
                shutil.copy(txtname, target_dir + "\\")
    print("copy file num: ", count)


# devide sounds into 4s segments
def pos_dir_make(dir_path, patient_id, pos):
    for po in pos:
        subdir = dir_path + patient_id + "\\" + patient_id + po
        wavname = subdir + ".wav"
        if os.path.exists(wavname):
            print("exist")
            mkdir(subdir)  # make dir


def index_load(tsvname):
    """读取tsv文件内容,不需要close函数"""
    with open(tsvname, "r") as f:
        txt_data = f.read()
    head = ["start", "end", "period"]
    data = txt_data.split("\n")[:-1]
    # 遍历每一行
    for l in data:
        sgmt = l.split("\t")
        if sgmt[2] != "0":
            head = np.vstack([head, sgmt])
    return head[1:]


# preprocessed PCGs were segmented into four heart sound states
def period_div(
    path,
    mur,
    patient_id_list,
    positoin,
    id_data,
    Murmur_locations,
    Systolic_murmur_timing,
    Diastolic_murmur_timing,
):

    for patient_id in patient_id_list:
        patient_dir_path = path + mur + patient_id + "\\" + patient_id
        txtpath = patient_dir_path+".txt"
        current_patient_data = load_patient_data(txtpath)
        hunman_feat = get_features_mod(current_patient_data)
        for pos in positoin:
            dir_path = path + mur + patient_id + "\\" + patient_id + pos
            tsv_path = dir_path + ".tsv"
            wav_path = dir_path + ".wav"
            index = id_data.index(patient_id)
            wav_location = pos[1:]  # 听诊区域
            locations = Murmur_locations[index].split("+")  # 有杂音的区域
            # 此听诊区有杂音
            if wav_location in locations:
                murmur_type = "Present"
                Systolic_state = Systolic_murmur_timing[index]
                Diastolic_state = Diastolic_murmur_timing[index]
                # 没有 Systolic murmur
                Systolic_murmur = "Absent" if Systolic_state == "nan" else "Present"
                # 没有 Diastolic murmur
                Diastolic_murmur = "Absent" if Diastolic_state == "nan" else "Present"
            # 此听诊区没有杂音
            else:
                murmur_type = "Absent"
                Systolic_murmur = "Absent"
                Diastolic_murmur = "Absent"
                Systolic_state = "nan"
                Diastolic_state = "nan"
            # 如果是present的有杂音区域，或absent区域
            # if (mur == "Absent\\") or (mur == "Present\\" and (wav_location in locations)):
            if os.path.exists(tsv_path):
                state_div2(
                    tsv_path,
                    wav_path,
                    dir_path + "\\",
                    patient_id + pos,
                    murmur_type
                    # Systolic_murmur,
                    # Diastolic_murmur,
                    # Systolic_state,
                    # Diastolic_state,
                    # hunman_feat
                )


def state_div(
    tsvname,
    wavname,
    state_path,
    index,
    Systolic_murmur,
    Diastolic_murmur,
    Systolic_state,
    Diastolic_state,
    hunman_feat
):
    """切割出s1+收缩和s2+舒张"""
    index_file = index_load(tsvname)
    recording, fs = librosa.load(wavname, sr=4000)
    num = 0
    # start_index1 = 0
    # end_index1 = 0
    # start_index2 = 0
    # end_index2 = 0
    # count = 0
    for i in range(index_file.shape[0] - 3):
        # if count == 20:
        #     break
        if index_file[i][2] == "1" and index_file[i + 2][2] == "3":
            start_index1 = float(index_file[i][0]) * fs
            end_index1 = float(index_file[i+1][1]) * fs
            start_index2 = float(index_file[i + 2][0]) * fs
            end_index2 = float(index_file[i + 3][1]) * fs
            num = num + 1

            #  解决出现_0.wav的问题
            print(start_index1, end_index1, start_index2, end_index2)
            print("=============================================")
            print("wav name: " + wavname)
            buff1 = recording[int(start_index1): int(end_index1)]  # 字符串索引切割
            buff2 = recording[int(start_index2): int(end_index2)]  # 字符串索引切割
            print("buff1 len: " + str(len(buff1)),
                  "buff2 len: " + str(len(buff2)))
            # if Systolic_murmur == "Present" and Diastolic_murmur == "Absent":
            #     # 切收缩期
            #     soundfile.write(
            #         state_path
            #         + f"{index}_s1+Systolic_{num}_{Systolic_murmur}_{Systolic_state}_{hunman_feat}.wav",
            #         buff1,
            #         fs,
            #     )
            # else:
            # 切收缩期
            soundfile.write(
                state_path
                + f"{index}_s1+Systolic_{num}_{Systolic_murmur}_{Systolic_state}_{hunman_feat}.wav",
                buff1,
                fs,
            )
            # 切舒张期
            soundfile.write(
                state_path
                + f"{index}_s2+Diastolic_{num}_{Diastolic_murmur}_{Diastolic_state}_{hunman_feat}.wav",
                buff2,
                fs,
            )
            # count += 1


def state_div2(
    tsvname,
    wavname,
    state_path,
    id_pos,
    murmur_type

):
    """按照4s切片"""
    index_file = index_load(tsvname)
    spilt_len = 4  # 切割长度为spilt_len s
    recording, fs = librosa.load(wavname, sr=4000)
    start = float(index_file[0][0]) * fs
    start = float(index_file[-1][1]) * fs
    buff = recording[int(start): int(start)]  # 要切割的数据
    # 将buff数据切割为3s的数据
    # 计算音频的总长度（秒）
    # total_length = len(recording) / fs

    # 计算每个片段的样本数
    samples_per_segment = spilt_len * fs

    # 切割音频数据
    segments = []
    for start in np.arange(0, len(recording), samples_per_segment):
        end = start + samples_per_segment
        segment = recording[start:end]
        segments.append(segment)

    segments.pop()
    for i, segment in enumerate(segments):
        soundfile.write(
            state_path
            + "{}_{}_{}_{}_{}.wav".format(
                id_pos, str(spilt_len)+"s", i, murmur_type, "none"
            ),
            segment,
            fs,
        )


# get patient id from csv file
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row["0"] for row in reader]  # weight 同列的数据
        return id


def fold_devide(data, flod_num=5):
    """五折交叉验证
    将输入列表打乱，然后分成五份
    output: flod5 = {0:[],1:[],2:[],3:[],4:[]}
    """
    # 打乱序列
    random.shuffle(data)
    # 五折
    flod5 = {}
    point = []
    for i in range(flod_num):
        point.append(i*round(len(data)/flod_num))
    # print(point)
    # 分割序列
    for i in range(len(point)):
        if i < len(point)-1:
            flod5[i] = []
            flod5[i].extend(data[point[i]:point[i+1]])
        else:
            flod5[i] = []
            flod5[i].extend(data[point[-1]:])
    return flod5

# copy data to folder


def copy_states_data(patient_id, folder, type, murmur):
    traget_path = folder+type+murmur
    if not os.path.exists(traget_path):
        os.makedirs(traget_path)
    for id in patient_id:
        dir_path = folder + murmur + id
        print(dir_path)
        for root, dir, file in os.walk(dir_path):
            for subdir in dir:
                subdir_path = os.path.join(root, subdir)
                print(subdir_path)
                if os.path.exists(dir_path):
                    shutil.copytree(subdir_path, traget_path + subdir)
                else:
                    print("dir not exist")


def data_set(root_path):
    """数据增强，包括时间拉伸和反转"""
    # root_path = r"D:\Shilong\murmur\01_dataset\06_new5fold"
    npy_path_padded = root_path+r"\npyFile_padded\npy_files01_norm"
    index_path = root_path + r"\npyFile_padded\index_files01_norm"
    if not os.path.exists(npy_path_padded):
        os.makedirs(npy_path_padded)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    for k in range(5):
        mel_list = []
        src_fold_root_path = root_path+r"\fold_set_"+str(k)
        # TODO 是否做数据增强
        # data_Auge(src_fold_root_path)
        for folder in os.listdir(src_fold_root_path):
            dataset_path = os.path.join(src_fold_root_path, folder)
            if k == 0 and folder == "absent":
                wav, label, names, index, data_id, feat = get_wav_data(
                    dataset_path, num=0)  # absent
            else:
                wav, label, names, index, data_id, feat = get_wav_data(
                    dataset_path, data_id)  # absent
            for i in range(len(wav)):
                mel = get_logmel_feature(wav[i])
                mel_list.append(mel)
            np.save(npy_path_padded +
                    f"\\{folder}_mel_norm01_fold{k}.npy", mel_list)
            np.save(npy_path_padded +
                    f"\\{folder}_wav_norm01_fold{k}.npy", wav)
            np.save(npy_path_padded +
                    f"\\{folder}_labels_norm01_fold{k}.npy", label)
            np.save(npy_path_padded +
                    f"\\{folder}_index_norm01_fold{k}.npy", index)
            np.save(npy_path_padded +
                    f"\\{folder}_name_norm01_fold{k}.npy", names)
            np.save(npy_path_padded +
                    f"\\{folder}_feat_norm01_fold{k}.npy", feat)
            absent_train_dic = zip(index, names, feat)
            pd.DataFrame(absent_train_dic).to_csv(
                index_path+f"\\fold{k}_{folder}_disc.csv", index=False, header=False)
    print("data set is done!")


def get_wav_data(dir_path, num=0):
    """返回数据文件"""
    wav = []
    label = []
    file_names = []
    wav_nums = []
    feat = []
    # 设置采样率为4k，时间长度为4
    fs = 4000
    time = 4
    data_length = fs*time
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 序号
                num = num+1
                file_names.append(subfile)
                wav_nums.append(num)
                # 数据读取
                print("reading: " + subfile)
                y, sr = librosa.load(wav_path, sr=4000)
                # TODO 采样率:4k
                # y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                y_4k_norm = wav_normalize(y)  # 归一化
                # 数据裁剪
                if y_4k_norm.shape[0] < data_length:
                    y_4k_norm = np.pad(
                        y_4k_norm,
                        ((0, data_length - y_4k_norm.shape[0])),
                        "constant",
                        constant_values=(0, 0),
                    )
                elif y_4k_norm.shape[0] > data_length:
                    # y_4k_norm = y_4k_norm[-data_length:]
                    y_4k_norm = y_4k_norm[:data_length]
                print("num is "+str(num), "y_16k size: "+str(y_4k_norm.size))

                wav.append(y_4k_norm)
                file_name = subfile.split("_")
                # 标签读取
                if file_name[4] == "Absent":  # Absent
                    label.append(0)
                elif file_name[4] == "Present":  # Present
                    label.append(1)  # 说明该听诊区有杂音
                feat.append(file_name[-1])

    return wav, label, file_names, wav_nums, num, feat


def wav_normalize(data):
    """min max归一化"""
    # range = np.max(data) - np.min(data)
    data = (data-np.mean(data))/np.max(np.abs(data))
    # data = (data-np.min(data))/range
    return data
    # recording -= recording.mean()
    # recording /= recording.abs().max()
