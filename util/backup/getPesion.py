import csv

import numpy as np


# 先去掉分错的idx


def csv_reader_cl(file_name, clo_num):
    """read csv file by column
    """
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


error_idx_path = r"C:\Users\611.ext\Desktop\idx\AudioClassifier\fold3.csv"
error_idx = csv_reader_cl(error_idx_path, 0)
present_idx_path = r"C:\Users\611.ext\Desktop\idx\fold3_present_disc.csv"
absnet_idx_path = r"C:\Users\611.ext\Desktop\idx\fold3_absent_disc.csv"
present_idx = csv_reader_cl(present_idx_path, 0)
present_names = csv_reader_cl(present_idx_path, 1)
absent_idx = csv_reader_cl(absnet_idx_path, 0)
absent_names = csv_reader_cl(absnet_idx_path, 1)
all_idx = present_idx + absent_idx
correct_idx = list(set(all_idx) - set(error_idx))
correct_names = []
for idx in correct_idx:
    if idx in present_idx:
        correct_names.append(present_names[present_idx.index(idx)])
    # else:
    #     correct_names.append(absent_names[absent_idx.index(idx)])
# print(len(correct_idx))
# print((correct_names))

# 然后解析所有分对的idx
location = []
locations_murmur = {}
for name in correct_names:
    name_string = name.split("_")
    key = name_string[0] + "_" + name_string[1]
    period = "s1" if (name_string[2] == "s1+Systolic") else "s2"
    murmur = 0 if (name_string[4] == "Absent") else 1
    if key not in locations_murmur:
        locations_murmur[key] = {}
    if period not in locations_murmur[key]:
        locations_murmur[key][period] = []
    locations_murmur[key][period].append(murmur)
# 按照0.5的阈值判定每个听诊区是收缩期杂音还是舒张期杂音
result_dic = {}
for key in locations_murmur:
    if key not in result_dic:
        result_dic[key] = {}
    if period not in locations_murmur[key]:
        result_dic[key][period] = 0
    for period in locations_murmur[key]:
        result_dic[key][period] = 1 if np.mean(
            locations_murmur[key][period]) > 0.5 else 0
# result_dic.pop('50159_MV')
result_dic2 = {}
for loc in result_dic:
    result_dic2[loc] = None
    if result_dic[loc].get('s1', 0) == 1 and result_dic[loc].get('s2', 0) == 1:
        result_dic2[loc] = 3
    elif result_dic[loc].get('s1', 0) == 1 and result_dic[loc].get('s2', 0) == 0:
        result_dic2[loc] = 2
    elif result_dic[loc].get('s1', 0) == 0 and result_dic[loc].get('s2', 0) == 1:
        result_dic2[loc] = 1
    elif result_dic[loc].get('s1', 0) == 0 and result_dic[loc].get('s2', 0) == 0:
        result_dic2[loc] = 'None'

# 按照听诊区的杂音类型，判定每个患者的杂音类型
patient_id_dic = {}
for id in locations_murmur:
    patient_id = id.split("_")[0]
    if patient_id not in patient_id_dic:
        patient_id_dic[patient_id] = {}
    if id not in patient_id_dic[patient_id]:
        patient_id_dic[patient_id][id] = result_dic2[id]

patient_result = {}
for pid in patient_id_dic:
    # 得到所有present的定位结果
    patient_result[pid] = list(set(patient_id_dic[pid].values()))

print(patient_result)
