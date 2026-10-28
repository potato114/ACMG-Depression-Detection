#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 Depression_get_dataset_csv.py
* @Time 	:	 2024/01/13
* @Author	:	 xcq
* @Version	:	 1.0
* @Contact	:	 cq.xu@siat.ac.cn
* @License	:	 (C)Copyright 2023-2025, xcq&AISML
* @Desc   	:	 获取训练所用的csv文件,内容包含训练集测试集名单和相应label
'''


import os
import csv
import pandas as pd


def _get_fold_for_subject_id(excel_path, subject_id):
  # 读取Excel文件
  df = pd.read_excel(excel_path)

  # 根据Directory筛选对应的Fold
  fold_for_directory = df.loc[df['Directory'] == subject_id, 'Fold'].unique()

  if len(fold_for_directory) == 1:
    return fold_for_directory[0]
  else:
    return None


def get_folds_csv(resample_segments_directory, resample_segments_folds_csv_dir, folds_excel_path, fold_id):
  wav_data = {}

  for subdir, _, files in os.walk(os.path.join(resample_segments_directory)):
    for wav_file in files:
      if wav_file.endswith('.wav'):
        # 获得lable
        if 'Normal' in wav_file:
          label = 'Normal'
        elif 'Depression' in wav_file:
          label = 'Mild'
        elif 'Moderate' in wav_file:
          label = 'Moderate'
        # 获得segment_id,speech_label
        wav_file_path = os.path.join(subdir, wav_file)
        segment_id, speech_label = parse_filename(wav_file)
        subject_id = segment_id.split('_')[0]
        wav_fold = _get_fold_for_subject_id(folds_excel_path, subject_id)

        if fold_id == wav_fold:
          state = 'test'
        else:
          state = 'train'
        wav_data[segment_id] = {'segment_id': segment_id, 'label':label, 'speech_label': speech_label, 'state':state}

  # 创建目标文件夹
  os.makedirs(resample_segments_folds_csv_dir, exist_ok = True)
  # 输出csv文件路径
  output_csv = resample_segments_folds_csv_dir + '/fold_' + str(fold_id) + '.csv'

  # Process .txt files
  with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['segment_id', 'label', 'speech_label', 'txt_label', 'state']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()

    for subdir, _, files in os.walk(os.path.join(resample_segments_directory)):
      for txt_file in files:
        if txt_file.endswith('.txt'):
          txt_file_path = os.path.join(subdir, txt_file)
          segment_id, txt_label = parse_filename(txt_file)

          if segment_id in wav_data:
            writer.writerow({'segment_id': segment_id, 'label':wav_data[segment_id]['label'], 'speech_label': wav_data[segment_id]['speech_label'], 'txt_label': txt_label, 'state': wav_data[segment_id]['state']})

  print(f'{output_csv}已输出！！！')

def parse_filename(filename):
  parts = filename.rsplit('_', 2)
  segment_id = '_'.join(parts[:-1])
  label = parts[-1].split('.')[0]  # Remove file extension
  return segment_id, label


if __name__ == "__main__":

  resample_segments_directory = 'data/dataset/resample_segments'
  resample_segments_folds_csv_dir = 'data/dataset/resample_segments_folds_csv'
  folds_excel_path = 'data/dataset_info/fold.xlsx'

  # 五折获得五个csv文件
  for i in range(1, 6):
    get_folds_csv(resample_segments_directory, resample_segments_folds_csv_dir, folds_excel_path, i)

  print('All Finished!!!')
