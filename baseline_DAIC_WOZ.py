#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 main.py
* @Time 	:	 2024/01/20
* @Author	:	 yhb
* @Version	:	 1.0
* @Contact	:	 hb.yu@siat.ac.cn
* @Copyright	:	 Copyright 2025-2026, yhb&AISML
* SPDX-License-Identifier: Apache-2.0
* @Desc   	:	 语音和本文一致性分析main函数入口
'''


import yaml
import torch
import logging
import torch.optim as optim
import argparse
from pathlib import Path
from argparse import Namespace
from utils import TrainTool_baseline_2class as tt
from utils import create_workshop as cw
from torch.optim.lr_scheduler import CosineAnnealingLR


def run(config):

  # 获得模型
  model = tt.get_model(config)
  
  # 获得train_dataloader, test_dataloader
  train_dataloader, test_dataloader = tt.get_dataloaders(config)
  
  # 定义损失函数和优化器
  criterion = tt.build_criterion(config)
  # Optimizer (support AdamW + optional weight_decay from config).
  optim_name = str(getattr(config, "optim", "Adam"))
  weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
  optim_kwargs = {"lr": config.lr}
  if weight_decay > 0:
    optim_kwargs["weight_decay"] = weight_decay
  optimizer = optim.__dict__[optim_name](model.parameters(), **optim_kwargs)

  # 用于学习率衰减
  scheduler = CosineAnnealingLR(optimizer, T_max = 30, eta_min = 0.00000001)

  # 配置日志
  logging.basicConfig(filename = config.exp_path + '/log/training.log', 
  level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', filemode = 'a')
  logging.info(f"------------------------起始线-----------------------------------")

  # 早停，防止过拟合
  best_test_loss = float('inf')    # 初始设定一个无穷大的值作为最佳损失
  patience = config.early_stop_patience   # 允许连续多少轮测试集损失不降低
  patience_counter = 0

  # 初始化最佳测试集subject_f1分数，用于保存最好的模型
  best_test_subject_f1 = 0.0

  # 存储训练和测试指标的字典,用于画图
  metrics_dict_hard_vote = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],

    "train_segment_accuracy": [],
    "train_segment_f1": [],
    "train_segment_precision": [],
    "train_segment_recall": [],

    "train_subject_accuracy": [],
    "train_subject_f1": [],
    "train_subject_precision": [],
    "train_subject_recall": [],

    "test_segment_accuracy": [],
    "test_segment_f1": [],
    "test_segment_precision": [],
    "test_segment_recall": [],

    "test_subject_accuracy": [],
    "test_subject_f1": [],
    "test_subject_precision": [],
    "test_subject_recall": []
  }
  # 存储训练和测试指标的字典,用于画图
  metrics_dict_soft_vote = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],

    "train_segment_accuracy": [],
    "train_segment_f1": [],
    "train_segment_precision": [],
    "train_segment_recall": [],

    "train_subject_accuracy": [],
    "train_subject_f1": [],
    "train_subject_precision": [],
    "train_subject_recall": [],

    "test_segment_accuracy": [],
    "test_segment_f1": [],
    "test_segment_precision": [],
    "test_segment_recall": [],

    "test_subject_accuracy": [],
    "test_subject_f1": [],
    "test_subject_precision": [],
    "test_subject_recall": []
  }


  # 模型训练
  for epoch in range(config.epochs):
    # 训练
    model, train_loss, train_segment_performance, train_subject_performance_hard_vote, train_subject_performance_soft_vote = tt.train_epoch(model, train_dataloader, optimizer, criterion, config.device, epoch, config)
    
    # 测试
    test_loss, test_segment_performance, test_subject_performance_hard_vote, test_subject_performance_soft_vote = tt.test_epoch(model, test_dataloader, criterion, config.device, epoch, config)
    train_loss = train_loss / len(train_dataloader)
    test_loss = test_loss / len(test_dataloader)
    # 学习率衰减
    scheduler.step()

    # 检查测试best_test_subject_f1是否提高，如果提高则保存模型
    if test_subject_performance_hard_vote["f1"] > best_test_subject_f1:
      best_test_subject_f1 = test_subject_performance_hard_vote["f1"]
      torch.save(
        {
          'epoch': epoch,
          'model_type': config.model_type,
          'model_config': config.model_config[config.model_type],
          'state_dict': model.state_dict(),
        },
        f"{config.exp_path}/model/best_model.pth"
      )
    
    # 用于画图
    metrics_dict_hard_vote["epoch"].append(epoch)
    metrics_dict_hard_vote["train_loss"].append(train_loss)
    metrics_dict_hard_vote["test_loss"].append(test_loss)
    metrics_dict_hard_vote["train_segment_accuracy"].append(train_segment_performance["accuracy"])
    metrics_dict_hard_vote["train_segment_f1"].append(train_segment_performance["f1"])
    metrics_dict_hard_vote["train_segment_precision"].append(train_segment_performance["precision"])
    metrics_dict_hard_vote["train_segment_recall"].append(train_segment_performance["recall"])
    metrics_dict_hard_vote["train_subject_accuracy"].append(train_subject_performance_hard_vote["accuracy"])
    metrics_dict_hard_vote["train_subject_f1"].append(train_subject_performance_hard_vote["f1"])
    metrics_dict_hard_vote["train_subject_precision"].append(train_subject_performance_hard_vote["precision"])
    metrics_dict_hard_vote["train_subject_recall"].append(train_subject_performance_hard_vote["recall"])
    metrics_dict_hard_vote["test_segment_accuracy"].append(test_segment_performance["accuracy"])
    metrics_dict_hard_vote["test_segment_f1"].append(test_segment_performance["f1"])
    metrics_dict_hard_vote["test_segment_precision"].append(test_segment_performance["precision"])
    metrics_dict_hard_vote["test_segment_recall"].append(test_segment_performance["recall"])
    metrics_dict_hard_vote["test_subject_accuracy"].append(test_subject_performance_hard_vote["accuracy"])
    metrics_dict_hard_vote["test_subject_f1"].append(test_subject_performance_hard_vote["f1"])
    metrics_dict_hard_vote["test_subject_precision"].append(test_subject_performance_hard_vote["precision"])
    metrics_dict_hard_vote["test_subject_recall"].append(test_subject_performance_hard_vote["recall"])
    cw.draw_exp_result(metrics_dict_hard_vote, config, 'hard_vote')

    metrics_dict_soft_vote["epoch"].append(epoch)
    metrics_dict_soft_vote["train_loss"].append(train_loss)
    metrics_dict_soft_vote["test_loss"].append(test_loss)
    metrics_dict_soft_vote["train_segment_accuracy"].append(train_segment_performance["accuracy"])
    metrics_dict_soft_vote["train_segment_f1"].append(train_segment_performance["f1"])
    metrics_dict_soft_vote["train_segment_precision"].append(train_segment_performance["precision"])
    metrics_dict_soft_vote["train_segment_recall"].append(train_segment_performance["recall"])
    metrics_dict_soft_vote["train_subject_accuracy"].append(train_subject_performance_soft_vote["accuracy"])
    metrics_dict_soft_vote["train_subject_f1"].append(train_subject_performance_soft_vote["f1"])
    metrics_dict_soft_vote["train_subject_precision"].append(train_subject_performance_soft_vote["precision"])
    metrics_dict_soft_vote["train_subject_recall"].append(train_subject_performance_soft_vote["recall"])
    metrics_dict_soft_vote["test_segment_accuracy"].append(test_segment_performance["accuracy"])
    metrics_dict_soft_vote["test_segment_f1"].append(test_segment_performance["f1"])
    metrics_dict_soft_vote["test_segment_precision"].append(test_segment_performance["precision"])
    metrics_dict_soft_vote["test_segment_recall"].append(test_segment_performance["recall"])
    metrics_dict_soft_vote["test_subject_accuracy"].append(test_subject_performance_soft_vote["accuracy"])
    metrics_dict_soft_vote["test_subject_f1"].append(test_subject_performance_soft_vote["f1"])
    metrics_dict_soft_vote["test_subject_precision"].append(test_subject_performance_soft_vote["precision"])
    metrics_dict_soft_vote["test_subject_recall"].append(test_subject_performance_soft_vote["recall"])
    cw.draw_exp_result(metrics_dict_soft_vote, config, 'soft_vote')
    
    # 保存到log中
    logging.info(f"超参数: lr:{config.lr}, batch_size:{config.batch_size * config.accum_iter}, epochs:{config.epochs}, optimizer:{config.optim}, seed:{config.seed}  \n \
      - Epoch [{epoch}/{config.epochs - 1}]  \n \
      ====================训练集====================  \n \
        - Loss: {train_loss:.4f}  \n\n \
        ---------------Segment Level---------------  \n \
        - Accuracy: {100 * train_segment_performance['accuracy']:.4f}%  - F1: {train_segment_performance['f1']:.4f}  \n \
        - Precision: {train_segment_performance['precision']:.4f}  - Recall: {train_segment_performance['recall']:.4f}  \n\n \
        ---------------Subject Level Hard Vote---------------  \n \
        - Accuracy: {100 * train_subject_performance_hard_vote['accuracy']:.4f}%  - F1: {train_subject_performance_hard_vote['f1']:.4f}  \n \
        - Precision: {train_subject_performance_hard_vote['precision']:.4f}  - Recall: {train_subject_performance_hard_vote['recall']:.4f}  \n\n \
        ---------------Subject Level Soft Vote---------------  \n \
        - Accuracy: {100 * train_subject_performance_soft_vote['accuracy']:.4f}%  - F1: {train_subject_performance_soft_vote['f1']:.4f}  \n \
        - Precision: {train_subject_performance_soft_vote['precision']:.4f}  - Recall: {train_subject_performance_soft_vote['recall']:.4f}  \n\n \
      ====================测试集====================  \n \
        - Loss: {test_loss:.4f}  \n\n \
        ---------------Segment Level---------------  \n \
        - Accuracy: {100 * test_segment_performance['accuracy']:.4f}%  - F1: {test_segment_performance['f1']:.4f}  \n \
        - Precision: {test_segment_performance['precision']:.4f}  - Recall: {test_segment_performance['recall']:.4f}  \n\n \
        ---------------Subject Level Hard Vote---------------  \n \
        - Accuracy: {100 * test_subject_performance_hard_vote['accuracy']:.4f}%  - F1: {test_subject_performance_hard_vote['f1']:.4f}  \n \
        - Precision: {test_subject_performance_hard_vote['precision']:.4f}  - Recall: {test_subject_performance_hard_vote['recall']:.4f}  \n\n \
        ---------------Subject Level Soft Vote---------------  \n \
        - Accuracy: {100 * test_subject_performance_soft_vote['accuracy']:.4f}%  - F1: {test_subject_performance_soft_vote['f1']:.4f}  \n \
        - Precision: {test_subject_performance_soft_vote['precision']:.4f}  - Recall: {test_subject_performance_soft_vote['recall']:.4f}  \n\n \
      ")
    
    # 早停机制
    if config.early_stop:
      # 早停，防止过拟合
      if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0  # 重置耐心计数器
      else:
        patience_counter += 1
      if patience_counter >= patience:
        logging.info(f"Early stopping at epoch {epoch}.")
        break
  
  logging.info(f"------------------------终止线-----------------------------------\n\n\n\n\n")



def _resolve_path(project_path, value):
  path = Path(value).expanduser()
  return str(path if path.is_absolute() else project_path / path)


def parse_args():
  parser = argparse.ArgumentParser(description='Train a two-class model on DAIC-WOZ.')
  parser.add_argument('--config', default='config/config_DAIC.yaml')
  parser.add_argument('--device', default=None, help="For example: 'cuda:0' or 'cpu'.")
  parser.add_argument('--model-type', choices=['A_T', 'GateModel_A_T_bytime'], default=None)
  parser.add_argument('--seed', type=int, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  config_path = Path(args.config).expanduser().resolve()
  with config_path.open('r', encoding='utf-8') as file:
    config = Namespace(**yaml.safe_load(file))

  project_path = Path(__file__).resolve().parent
  config.project_path = str(project_path)
  config.model_type = args.model_type or config.model_type
  config.seed = args.seed if args.seed is not None else config.seed
  config.device = args.device or config.device
  config.dataset_csv = _resolve_path(project_path, config.fold_1_dataset_csv)
  config.speech_feature_data_dir = _resolve_path(project_path, config.speech_feature_data_dir)
  config.txt_feature_data_dir = _resolve_path(project_path, config.txt_feature_data_dir)

  cw.set_seed(config.seed)
  exp_mark = (
    f'seed_{config.seed}_lr_{config.lr}_'
    f'bs{config.batch_size * config.accum_iter}'
  )
  config.exp_path = str(
    project_path / config.output_dir / 'daic_woz' /
    config.model_type / 'fold_1' / exp_mark
  )
  cw.create_workshop(config.exp_path)
  run(config)
  print('All Finished!')
