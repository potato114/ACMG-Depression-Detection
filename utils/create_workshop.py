import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


# 设置随机数种子
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


# 创建exp目录，用于保存实验数据
def create_workshop(base_dir):
  # Confusion_Matrix
  confusion_matrix_dir = os.path.join(base_dir, 'Confusion_Matrix')

  segment_level_dir = os.path.join(confusion_matrix_dir, 'Segment_Level')
  train_segment_dir = os.path.join(segment_level_dir, 'train')
  test_segment_dir = os.path.join(segment_level_dir, 'test')

  subject_level_hard_vote_dir = os.path.join(confusion_matrix_dir, 'Subject_Level_hard_vote')
  train_subject_hard_vote_dir = os.path.join(subject_level_hard_vote_dir, 'train')
  test_subject_hard_vote_dir = os.path.join(subject_level_hard_vote_dir, 'test')

  subject_level_soft_vote_dir = os.path.join(confusion_matrix_dir, 'Subject_Level_soft_vote')
  train_subject_soft_vote_dir = os.path.join(subject_level_soft_vote_dir, 'train')
  test_subject_soft_vote_dir = os.path.join(subject_level_soft_vote_dir, 'test')

  # Classification result information
  result_info = os.path.join(base_dir, 'Result_Information')
  train_result_info = os.path.join(result_info, 'train')
  test_result_info = os.path.join(result_info, 'test')

  # log
  log_dir = os.path.join(base_dir, 'log')

  # model
  model_dir = os.path.join(base_dir, 'model')

  # 创建目录
  # Confusion_Matrix
  os.makedirs(train_segment_dir, exist_ok = True)
  os.makedirs(test_segment_dir, exist_ok = True)
  os.makedirs(train_subject_hard_vote_dir, exist_ok = True)
  os.makedirs(test_subject_hard_vote_dir, exist_ok = True)
  os.makedirs(train_subject_soft_vote_dir, exist_ok = True)
  os.makedirs(test_subject_soft_vote_dir, exist_ok = True)

  # Classification result information
  os.makedirs(train_result_info, exist_ok = True)
  os.makedirs(test_result_info, exist_ok = True)

  # log
  os.makedirs(log_dir, exist_ok = True)

  # model
  os.makedirs(model_dir, exist_ok = True)


# 实验数据可视化
def draw_exp_result(metrics_dict, config, mark):
  epochs = metrics_dict["epoch"]

  # 二行五列十个子图
  metrics_to_plot = [
    "train_subject_accuracy", "train_subject_f1", "train_subject_precision", "train_subject_recall", "train_loss",
    "test_subject_accuracy", "test_subject_f1", "test_subject_precision", "test_subject_recall", "test_loss"
  ]
  # 图的大小以及子图位置调整
  plt.figure(figsize = (22, 8))
  plt.subplots_adjust(wspace = 0.15, hspace = 0.3, bottom = 0.2)

  for idx, metric_name in enumerate(metrics_to_plot, start = 1):
    # 不同颜色
    color = plt.cm.viridis(idx / len(metrics_to_plot))
    plt.subplot(2, 5, idx)
    plt.plot(epochs, metrics_dict[metric_name], label = metric_name, marker = 'o', color = color)
    plt.xlabel('epoch')
    plt.title(metric_name, fontweight = 'bold')

  plt.tight_layout()
  # 保存图片
  plt.savefig(f"{config.exp_path}/exp_result_{mark}.jpg", dpi = 600)
