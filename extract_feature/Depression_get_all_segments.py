#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 Depression_get_all_segments.py
* @Time 	:	 2024/07/11
* @Author	:	 xcq
* @Version	:	 1.0
* @Contact	:	 cq.xu@siat.ac.cn
* @License	:	 (C)Copyright 2023-2025, xcq&AISML
* @Desc   	:	 将总数据库所有片段导出
'''


import os
import re
import functions_XCQ as xcq

def is_number(s):
  try:
    float(s)  # 尝试转换为浮点数
    return True
  except ValueError:
    return False

def export_segments(source_folder, target_folder):
  # 创建目标文件夹
  os.makedirs(target_folder, exist_ok = True)

  # 遍历源文件夹中的wav文件
  file_id = 1
  for file_name in os.listdir(source_folder):
    if file_name.endswith(".wav"):
      print(file_name)
      # 文件名称(不包含后缀)
      file_base_name = file_name.split('.')[0]
      subject_id = file_base_name.split('_')[0] + file_base_name.split('_')[-1]
      # wav文件完整路径
      wav_file_path = os.path.join(source_folder, file_name)
      # textgrid文件完整路径
      textgrid_file_path = wav_file_path.split('.')[0] + '.TextGrid'
      # 创建文件夹
      output_folder = os.path.join(target_folder, subject_id)
      os.makedirs(output_folder, exist_ok = True)

      if 'Normal1' in file_name:
        tier_name = 'normal'
        # 将每个wav片段以及对应文本导出
        time_stamps_patient, texts, speech, txt = xcq.get_time_stamps_texts_speech_txt(textgrid_file_path, tier_name)
      elif 'Normal2' in file_name:
        tier_name = 'doctor'
        # 将每个wav片段以及对应文本导出
        time_stamps_patient, texts, speech, txt = xcq.normal2(textgrid_file_path, tier_name)
      elif 'Depression1' in file_name:
        tier_name = 'depression'
        # 将每个wav片段以及对应文本导出
        time_stamps_patient, texts, speech, txt = xcq.depression1(textgrid_file_path, tier_name)
      elif 'Depression2' in file_name:
        tier_name = 'patient'
        # 将每个wav片段以及对应文本导出
        time_stamps_patient, texts, speech, txt = xcq.get_time_stamps_texts_speech_txt(textgrid_file_path, tier_name)
      elif 'Moderate' in file_name:
        tier_name = 'patient'
        # 将每个wav片段以及对应文本导出
        time_stamps_patient, texts, speech, txt = xcq.get_time_stamps_texts_speech_txt(textgrid_file_path, tier_name)

      assert len(time_stamps_patient) // 2 == len(texts) == len(speech) == len(txt)

      # print(f'时间戳的个数为：{len(time_stamps_patient) / 2}, 文本个数为：{len(texts)},\n语音标签个数为: {len(speech)}, 文本标签个数为: {len(txt)}\n')

      S = 0
      for i in range(0, len(time_stamps_patient), 2):
        # print(f'当前序号：{S}, 文本为：{texts[S]}, \n语音标签为: {speech[S]}, 文本标签为: {txt[S]}\n')
        if S >= len(speech) or S >= len(txt):
          S = min(len(speech), S >= len(txt)) - 1
        if not is_number(speech[S]):
          speech[S] = '0'
        # 导出片段
        speech_emotion = int(speech[S])
        # print(f'转换前speech_emotion标签为: {speech_emotion}')
        if (speech_emotion < 0):
          emtion = 'neg'
        elif (speech_emotion == 0):
          emtion = 'neu'
        elif (speech_emotion > 0):
          emtion = 'pos'
        # print(f'speech_emotion标签为: {emtion}')

        wav_output_file = os.path.join(output_folder, subject_id + '_s' + str(S).zfill(4) + '_'+ emtion + '.wav')
        xcq.export_wav_segment_file(wav_file_path, time_stamps_patient[i], time_stamps_patient[i + 1], wav_output_file)

        # 导出文本
        txt_emotion = int(txt[S])
        # print(f'txt_emotion标签为: {txt_emotion}')
        if (txt_emotion < 0):
          emtion = 'neg'
        elif (txt_emotion == 0):
          emtion = 'neu'
        elif (txt_emotion > 0):
          emtion = 'pos'
        txt_output_file = os.path.join(output_folder, subject_id + '_s' + str(S).zfill(4) + '_'+ emtion + '.txt')
        cleaned_text = re.sub(r'[^\w\s]', '', texts[S])  # 去除标点符号和特殊字符
        final_text = re.sub(r'\s', '', cleaned_text)       # 去除空格
        xcq.export_text_to_file(final_text, txt_output_file)
        # print('--------------------------------------')
        S += 1

      print(f'第{file_id}个音频切分完毕！')
      file_id += 1


if __name__ == "__main__":

  source_folder = "data/raw/all_data_16k_wav"  # 所有wav文件
  target_folder = "data/dataset/all_segments"

  export_segments(source_folder, target_folder)

  print("All Finished!!!")
