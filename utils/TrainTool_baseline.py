#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 TrainTool.py
* @Time 	:	 2024/01/20
* @Author	:	 yhb
* @Version	:	 1.0
* @Contact	:	 hb.yu@siat.ac.cn
* @Copyright	:	 Copyright 2025-2026, yhb&AISML
* SPDX-License-Identifier: Apache-2.0
* @Desc   	:	 训练过程中的所使用到的函数集合
'''

import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# 重写Dataset
class DepressionDataset(Dataset):
  def __init__(self, data_dict, config):
    self.data_dict = data_dict
    self.config = config
    self.segment_ids = list(data_dict.keys())

  def pad_input(self, x, length, pad_value=0):
    """
    x: np.ndarray or torch.Tensor, shape [T, D]
    length: target length
    return:
        x_pad: torch.Tensor [length, D]
        mask: torch.BoolTensor [length], True = padding
    """

    # ---- 1. 统一类型为 torch.Tensor ----
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif torch.is_tensor(x):
        x = x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    T, D = x.shape
    device = x.device
    dtype = x.dtype

    # ---- 2. 构造 mask（True = padding）----
    mask = torch.zeros(length, device=device, dtype=torch.bool)

    # ---- 3. pad / truncate ----
    if T < length:
        pad_len = length - T
        pad = torch.full((pad_len, D), pad_value, dtype=dtype, device=device)
        x = torch.cat([x, pad], dim=0)
        mask[-pad_len:] = True
    else:
        x = x[:length, :]

    return x, mask


  def __getitem__(self, idx):
    segment_id = self.segment_ids[idx]
    data = self.data_dict[segment_id]

    item = {
      'segment_id': segment_id,
      'subject_id': data['subject_id'],
      'speech_feature': self.pad_input(data['speech_feature'], self.config.speech_length),
      'txt_feature': self.pad_input(data['txt_feature'], self.config.txt_length),
      'label': torch.tensor(data['label']),
      'consistence_labels': torch.tensor(data['consistence_labels'])
    }
    return item
  
  def __len__(self):
    return len(self.segment_ids)


# 获得train_dataloader, test_dataloader
def get_dataloaders(config):
  def _parse_subject_id(segment_id: str) -> str:
    # EATD segment_id example: Depression_t_3_negative_out_1 -> subject_id: Depression_t_3
    parts = str(segment_id).split('_')
    if (
      len(parts) >= 3
      and parts[0] in ('Depression', 'Normal')
      and parts[1] in ('t', 'v')
      and parts[2].isdigit()
    ):
      return '_'.join(parts[:3])
    return parts[0] if parts else str(segment_id)

  # 读取 CSV 文件
  csv_path = config.dataset_csv
  print(f'读取的csv为: {csv_path}')
  df = pd.read_csv(csv_path)
  
  # ===========处理训练集===========
  # 初始化字典
  train_data_dict = {}
  test_data_dict = {}

  # 遍历数据框中的每一行
  for index, row in df.iterrows():
    # if index >= 1000:  #用作调试
    #   break
    segment_id = row['segment_id']
    label = row['label']
    speech_label = row['speech_label']
    txt_label = row['txt_label']
    state = row['state']

    # 处理状态为 "train" 的数据
    if state == 'train':
      # 获取subject_id
      subject_id = segment_id.split('_')[0]
      # subject_id = _parse_subject_id(segment_id)
      
      # 获取语音特征
      speech_feature_path = config.speech_feature_data_dir + '/' + segment_id + '.pt'
      loaded_data = torch.load(speech_feature_path)  # loaded_data为字典类型，有segment_id、subject_id、feature三个字段
      assert segment_id == loaded_data['segment_id']
      assert subject_id == loaded_data['subject_id']
      speech_feature = loaded_data['feature']

      # 获取文本特征
      txt_feature_path = config.txt_feature_data_dir + '/' + segment_id + '.pt'
      loaded_data = torch.load(txt_feature_path)  # loaded_data为字典类型，有segment_id、subject_id、feature三个字段
      assert segment_id == loaded_data['segment_id']
      assert subject_id == loaded_data['subject_id']
      txt_feature = loaded_data['feature']

      # 将 Normal映射为 0, Depression 映射为 1
      label_mapping = {'Normal': 0, 'Mild': 1, 'Moderate': 2}
      label_value = label_mapping[label]

      # 判断 speech_label 和 txt_label 是否相同
      consistence_labels = 1 if speech_label == txt_label else 0
      
      # 将记录添加到字典中
      train_data_dict[segment_id] = {'subject_id': subject_id, 'speech_feature': speech_feature, 'txt_feature': txt_feature, 'label': label_value, 'consistence_labels': consistence_labels}
    
    # 处理状态为 "test" 的数据
    if state == 'test':
      # 获取subject_id
      # if segment_id.endswith('_speed1.1') or segment_id.endswith('_speed0.9'):
      #   continue
      # if segment_id.endswith('_pitch-1') or segment_id.endswith('_pitch+1'):
      #   continue
      subject_id = segment_id.split('_')[0]
      # subject_id = _parse_subject_id(segment_id)
      
      # 获取语音特征
      speech_feature_path = config.speech_feature_data_dir + '/' + segment_id + '.pt'
      loaded_data = torch.load(speech_feature_path)  # loaded_data为字典类型，有segment_id、subject_id、feature三个字段
      assert segment_id == loaded_data['segment_id']
      assert subject_id == loaded_data['subject_id']
      speech_feature = loaded_data['feature']

      # 获取文本特征
      txt_feature_path = config.txt_feature_data_dir + '/' + segment_id + '.pt'
      loaded_data = torch.load(txt_feature_path)  # loaded_data为字典类型，有segment_id、subject_id、feature三个字段
      assert segment_id == loaded_data['segment_id']
      assert subject_id == loaded_data['subject_id']
      txt_feature = loaded_data['feature']

      # 将 Normal映射为 0, Mild 映射为 1，Moderate 映射为 2
      label_mapping = {'Normal': 0, 'Mild': 1, 'Moderate': 2}
      label_value = label_mapping[label]

      # 判断 speech_label 和 txt_label 是否相同
      consistence_labels = 1 if speech_label == txt_label else 0
      
      # 将记录添加到字典中
      test_data_dict[segment_id] = {'subject_id': subject_id, 'speech_feature': speech_feature, 'txt_feature': txt_feature, 'label': label_value, 'consistence_labels': consistence_labels}

  train_dataset = DepressionDataset(train_data_dict, config)
  test_dataset = DepressionDataset(test_data_dict, config)
  
  # 得到DataLoader
  train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True,num_workers=4,
    pin_memory=True,
    persistent_workers=True)
  test_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False,num_workers=4,
    pin_memory=True,
    persistent_workers=True)

  return train_dataloader, test_dataloader 

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    logits: [B, C]
    targets: [B] LongTensor
    gamma: focusing parameter
    alpha: 类别权重, [C] tensor 或 None
    """
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
    pt = (probs * targets_one_hot).sum(dim=1)  # 正确类别概率
    loss = - ((1 - pt) ** gamma) * torch.log(pt + 1e-8)

    if alpha is not None:
        at = (targets_one_hot * alpha).sum(dim=1)
        loss = at * loss

    return loss.mean()
# 获得model
def get_model(config):
  if config.model_type not in config.model_config:
    raise ValueError(f"Missing model_config entry for {config.model_type!r}.")

  kwargs = dict(config.model_config[config.model_type])
  additional_kwargs = {
    'num_classes': config.num_classes,
    'input_dim': config.input_dim,
    'ffn_embed_dim': config.ffn_embed_dim, 
  }
  kwargs = {**kwargs, **additional_kwargs}

  if config.model_type == 'A_T':
    from models.baseline import A_T
    model = A_T(**kwargs)
  elif config.model_type == 'GateModel_A_T_bytime':
    from models.acmg import GateModel_A_T
    model = GateModel_A_T(**kwargs)
  elif config.model_type == 'GateModel_A_T_unimodal':
    from models.acmg_Unimodal import UnimodalGateModel_A_T
    model = UnimodalGateModel_A_T(**kwargs)
  else:
    raise ValueError(
      f"Unsupported model_type={config.model_type!r}. "
      "Choose 'A_T', 'GateModel_A_T_bytime', or 'GateModel_A_T_unimodal'."
    )

  model.to(config.device)
  print(f'已加载模型{config.model_type}')
  return model


# 模型训练
def train_epoch(model, train_dataloader, optimizer, criterion, device, epoch, config):

  model.train()  # 模型用于训练

  total_loss = 0  # 总loss

  accum_iter = config.accum_iter  # 用于梯度累加

  results_dict = {}  # 用于记录每个segment的分类结果信息

  # 模型训练
  pbar = tqdm(train_dataloader)
  for batch_index, batch in enumerate(pbar):
    # 设置可视化附加信息
    postfix_str = f"Epoch {epoch} / {config.epochs - 1}"
    pbar.set_postfix_str(postfix_str, refresh = True)

    # feature label 为dataset封装属性
    x_a = batch['speech_feature'][0].to(device)
    x_a_padding_mask = batch['speech_feature'][1].to(device)
    x_t = batch['txt_feature'][0].to(device)
    x_t_padding_mask = batch['txt_feature'][1].to(device)
    y = batch['label'].to(device)
    
    # forward
    out = model(x_a, x_t, x_a_padding_mask, x_t_padding_mask)  # 模型输出
    loss = criterion(out, y)  # 计算loss

    total_loss = total_loss + loss.item()   # 累加整个训练数据集的loss
    
    # backward
    loss = loss / accum_iter # 用于梯度累加
    loss.backward()

    # 用于梯度累加
    if ((batch_index + 1) % accum_iter == 0) or (batch_index + 1 == len(train_dataloader)):
      optimizer.step()
      optimizer.zero_grad()
    
    # 模型预测结果
    preds = torch.argmax(out, dim = 1)

    # 将每个segment的分类结果保存到results_dict
    for i in range(len(batch['segment_id'])):
      segment_id = batch['segment_id'][i]
      subject_id_value = batch['subject_id'][i]
      y_pre_p_class_0_value = torch.nn.functional.softmax(out, dim = 1)[i, 0].item()
      y_pre_p_class_1_value = torch.nn.functional.softmax(out, dim = 1)[i, 1].item()
      y_pre_p_class_2_value = torch.nn.functional.softmax(out, dim = 1)[i, 2].item()
      y_pred_value = preds[i].item()
      y_true_value = y[i].item()
      
      # 添加到字典
      results_dict[segment_id] = {
        'subject_id': subject_id_value,
        'y_pre_p_class_0': y_pre_p_class_0_value,
        'y_pre_p_class_1': y_pre_p_class_1_value,
        'y_pre_p_class_2': y_pre_p_class_2_value,
        'y_pred': y_pred_value,
        'y_true': y_true_value
      }

  pbar.close()

  state = 'train' # 标志为训练

  # 将分类结果写到csv文件中
  csv_save_path = f"{config.exp_path}/Result_Information/{state}/Epoch_{epoch}.csv"
  _write_results_to_csv(results_dict, csv_save_path)


  # ========== segment级别 ==========
  segment_true, segment_pred = _statistical_segment(results_dict)
  # 统计分类指标
  accuracy_segment, f1_segment, precision_segment, recall_segment = _statistical_performance_indicators(segment_true, segment_pred)
  segment_performance = {
    "accuracy": accuracy_segment,
    "f1": f1_segment,
    "precision": precision_segment,
    "recall": recall_segment}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = segment_true,			
    label_pred = segment_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Segment Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Segment_Level/{state}/Epoch_{epoch}.jpg",
    dpi = 300)


  # ========== subject级别 (硬投票)==========
  subject_hard_vote_true, subject_hard_vote_pred = _statistical_subject_hard_vote(results_dict)
  # 统计分类指标
  accuracy_subject, f1_subject, precision_subject, recall_subject = _statistical_performance_indicators(subject_hard_vote_true, subject_hard_vote_pred)
  subject_performance_hard_vote = {
    "accuracy": accuracy_subject,
    "f1": f1_subject,
    "precision": precision_subject,
    "recall": recall_subject}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = subject_hard_vote_true,			
    label_pred = subject_hard_vote_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Subject Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Subject_Level_hard_vote/{state}/Epoch_{epoch}.jpg",
    dpi = 300)


  # ========== subject级别 (软投票)==========
  subject_soft_vote_true, subject_soft_vote_pred = _statistical_subject_soft_vote(results_dict)
    # 统计分类指标
  accuracy_subject, f1_subject, precision_subject, recall_subject = _statistical_performance_indicators(subject_soft_vote_true, subject_soft_vote_pred)
  subject_performance_soft_vote = {
    "accuracy": accuracy_subject,
    "f1": f1_subject,
    "precision": precision_subject,
    "recall": recall_subject}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = subject_soft_vote_true,			
    label_pred = subject_soft_vote_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Subject Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Subject_Level_soft_vote/{state}/Epoch_{epoch}.jpg",
    dpi = 300)
  

  return model, total_loss, segment_performance, subject_performance_hard_vote, subject_performance_soft_vote


# 模型测试
def test_epoch(model, test_dataloader, criterion, device, epoch, config):

  model.eval()  # 模型用于测试

  total_loss = 0  # 总loss

  results_dict = {}  # 用于记录每个segment的分类结果信息
  
  # 模型测试
  pbar = tqdm(test_dataloader)
  for batch_index, batch in enumerate(pbar):
    # 设置附加信息
    postfix_str = f"Epoch {epoch} / {config.epochs - 1}"
    pbar.set_postfix_str(postfix_str, refresh = True)
    
    # feature label 为dataset封装属性
    x_a = batch['speech_feature'][0].to(device)
    x_a_padding_mask = batch['speech_feature'][1].to(device)
    x_t = batch['txt_feature'][0].to(device)
    x_t_padding_mask = batch['txt_feature'][1].to(device)
    y = batch['label'].to(device)
    
    # forward
    out = model(x_a, x_t, x_a_padding_mask, x_t_padding_mask)  # 模型输出
    loss = criterion(out, y)  # 计算loss
    total_loss = total_loss + loss.item()   # 累加整个训练数据集的loss

    # 模型预测结果
    preds = torch.argmax(out, dim = 1)

    # 将每个segment的分类结果保存到results_dict
    for i in range(len(batch['segment_id'])):
      segment_id = batch['segment_id'][i]
      subject_id_value = batch['subject_id'][i]
      y_pre_p_class_0_value = torch.nn.functional.softmax(out, dim = 1)[i, 0].item()
      y_pre_p_class_1_value = torch.nn.functional.softmax(out, dim = 1)[i, 1].item()
      y_pre_p_class_2_value = torch.nn.functional.softmax(out, dim = 1)[i, 2].item()
      y_pred_value = preds[i].item()
      y_true_value = y[i].item()
      
      # 添加到字典
      results_dict[segment_id] = {
        'subject_id': subject_id_value,
        'y_pre_p_class_0': y_pre_p_class_0_value,
        'y_pre_p_class_1': y_pre_p_class_1_value,
        'y_pre_p_class_2': y_pre_p_class_2_value,
        'y_pred': y_pred_value,
        'y_true': y_true_value
      }
  
  pbar.close()

  state = 'test' # 标志为训练

  # 将分类结果写到csv文件中
  csv_save_path = f"{config.exp_path}/Result_Information/{state}/Epoch_{epoch}.csv"
  _write_results_to_csv(results_dict, csv_save_path)


  # ========== segment级别 ==========
  segment_true, segment_pred = _statistical_segment(results_dict)
  # 统计分类指标
  accuracy_segment, f1_segment, precision_segment, recall_segment = _statistical_performance_indicators(segment_true, segment_pred)
  segment_performance = {
    "accuracy": accuracy_segment,
    "f1": f1_segment,
    "precision": precision_segment,
    "recall": recall_segment}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = segment_true,			
    label_pred = segment_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Segment Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Segment_Level/{state}/Epoch_{epoch}.jpg",
    dpi = 300)


  # ========== subject级别 (硬投票)==========
  subject_hard_vote_true, subject_hard_vote_pred = _statistical_subject_hard_vote(results_dict)
  # 统计分类指标
  accuracy_subject, f1_subject, precision_subject, recall_subject = _statistical_performance_indicators(subject_hard_vote_true, subject_hard_vote_pred)
  subject_performance_hard_vote = {
    "accuracy": accuracy_subject,
    "f1": f1_subject,
    "precision": precision_subject,
    "recall": recall_subject}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = subject_hard_vote_true,			
    label_pred = subject_hard_vote_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Subject Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Subject_Level_hard_vote/{state}/Epoch_{epoch}.jpg",
    dpi = 300)


  # ========== subject级别 (软投票)==========
  subject_soft_vote_true, subject_soft_vote_pred = _statistical_subject_soft_vote(results_dict)
    # 统计分类指标
  accuracy_subject, f1_subject, precision_subject, recall_subject = _statistical_performance_indicators(subject_soft_vote_true, subject_soft_vote_pred)
  subject_performance_soft_vote = {
    "accuracy": accuracy_subject,
    "f1": f1_subject,
    "precision": precision_subject,
    "recall": recall_subject}
  # 画混淆矩阵
  _draw_confusion_matrix(label_true = subject_soft_vote_true,			
    label_pred = subject_soft_vote_pred,
    labels_name = ["Normal", "Mild", "Moderate"],
    title = "Subject Level Confusion Matrix",
    pdf_save_path = f"{config.exp_path}/Confusion_Matrix/Subject_Level_soft_vote/{state}/Epoch_{epoch}.jpg",
    dpi = 300)

  
  return total_loss, segment_performance, subject_performance_hard_vote, subject_performance_soft_vote


# 将分类结果信息写入csv文件
def _write_results_to_csv(results_dict, csv_filename):
  # 将字典数据转换为列表，以便写入CSV文件
  rows = []
  for segment_id, values in results_dict.items():
    row = [segment_id]
    row.append(values['subject_id'])
    row.append(values['y_pre_p_class_0'])
    row.append(values['y_pre_p_class_1'])
    row.append(values['y_pre_p_class_2'])
    row.append(values['y_pred'])
    row.append(values['y_true'])
    rows.append(row)

  # 写入CSV文件
  with open(csv_filename, mode = 'w', newline = '') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入表头
    header = ['segment_id', 'subject_id', 'y_pre_p_class_0', 'y_pre_p_class_1', 'y_pre_p_class_2', 'y_pred', 'y_true']
    csv_writer.writerow(header)
    # 写入数据
    csv_writer.writerows(rows)


# 统计segment级别的结果
def _statistical_segment(results_dict):
  segment_true = []
  segment_pred = []

  for segment_id, values in results_dict.items():
    # 将当前 segment 的 y_true 和 y_pred 加入列表
    segment_true.append(values['y_true'])
    segment_pred.append(values['y_pred'])

  return segment_true, segment_pred


# 统计subject级别的结果(硬投票)
def _statistical_subject_hard_vote(results_dict):
  # Group per-subject predictions/labels.
  subject_preds = defaultdict(list)
  subject_trues = defaultdict(list)
  for _, values in results_dict.items():
    sid = values['subject_id']
    subject_preds[sid].append(int(values['y_pred']))
    subject_trues[sid].append(int(values['y_true']))

  def _majority(xs):
    # Deterministic tie-break: choose the smallest label among top-count.
    c = Counter(xs)
    best_cnt = max(c.values())
    return min([k for k, v in c.items() if v == best_cnt])

  subject_hard_vote_true = []
  subject_hard_vote_pred = []
  for sid in subject_preds.keys():
    subject_hard_vote_true.append(_majority(subject_trues[sid]))
    subject_hard_vote_pred.append(_majority(subject_preds[sid]))
  return subject_hard_vote_true, subject_hard_vote_pred


# 统计subject级别的结果(软投票)
def _statistical_subject_soft_vote(results_dict):
  subject_soft_vote_true = []
  subject_soft_vote_pred = []

  subject_trues = defaultdict(list)
  subject_probs_accumulator = defaultdict(lambda: [0.0, 0.0, 0.0])

  for _, values in results_dict.items():
    sid = values['subject_id']
    subject_trues[sid].append(int(values['y_true']))
    probs = [values['y_pre_p_class_0'], values['y_pre_p_class_1'], values['y_pre_p_class_2']]
    subject_probs_accumulator[sid] = [acc + float(p) for acc, p in zip(subject_probs_accumulator[sid], probs)]

  def _majority(xs):
    c = Counter(xs)
    best_cnt = max(c.values())
    return min([k for k, v in c.items() if v == best_cnt])

  for sid in subject_probs_accumulator.keys():
    subject_soft_vote_true.append(_majority(subject_trues[sid]))
    accumulated = subject_probs_accumulator[sid]
    max_class = max(range(len(accumulated)), key=lambda i: accumulated[i])
    subject_soft_vote_pred.append(int(max_class))
  return subject_soft_vote_true, subject_soft_vote_pred


# 获得分类指标
def _statistical_performance_indicators(y_true, y_pred):
  # 准确率
  accuracy = accuracy_score(y_true, y_pred)

  # 宏平均F1
  f1 = f1_score(y_true, y_pred, average = 'macro', zero_division = 0)

  # 宏平均precision
  precision = precision_score(y_true, y_pred, average = 'macro', zero_division = 0)

  # 宏平均recll
  recll = recall_score(y_true, y_pred, average = 'macro', zero_division = 0)

  return accuracy, f1, precision, recll


# 画混淆矩阵
def _draw_confusion_matrix(label_true, label_pred, labels_name, title = "Confusion Matrix", pdf_save_path = None, thresh = 0.8, dpi = 100, axis_labels = None):
  # 利用sklearn中的函数生成混淆矩阵并归一化
  cm = confusion_matrix(y_true = label_true, y_pred = label_pred)  # 生成混淆矩阵

  # 保存confusion_matrix数据以供统计使用
  confusion_matrix_save_path = pdf_save_path[:-4] + '.txt'
  # 保存混淆矩阵到文本文件
  np.savetxt(confusion_matrix_save_path, cm, fmt = "%d", delimiter = "\t")

  # 归一化
  percent = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]  

  # 自定义渐变色
  colors = ['#F8E0C6', '#D79E8C', '#B53A23', '#892819']  # 原神可莉配色
  custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N = 128)

  plt.figure(figsize = (8, 7))
  im = plt.imshow(cm, interpolation = 'nearest', cmap = custom_cmap)
  plt.colorbar(im, fraction = 0.046, pad = 0.04)  # 绘制图例
  # 图像标题
  if title is not None:
    plt.title(title)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.yticks(range(labels_name.__len__()), labels_name)
  plt.xticks(range(labels_name.__len__()), labels_name, rotation = 45)

  # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
  for i in range(np.shape(cm)[0]):
    for j in range(np.shape(cm)[1]):
      tmp = int(percent[i][j] * 100 + 0.5)
      if tmp >= 0:
        text = str(cm[i][j]) + "(" + format(tmp, 'd') + '%)'
        plt.text(j, i, text, ha = "center", va = "center",
        color = "white" if percent[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
  # 显示
  plt.savefig(pdf_save_path, bbox_inches = 'tight', dpi = dpi)
  plt.close()
    
