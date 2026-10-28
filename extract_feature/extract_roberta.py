import os
import torch
import numpy as np
from fairseq.models.roberta import RobertaModel


def extract_roberta(model, txtfile, savefile, segment_id, subject_id, window_size = 512, stride = 512):
  with open(txtfile, 'r') as f:
    text = f.read()

  tokens = model.encode(text)
  embedding_list = []

  for start in range(0, len(tokens), stride):
    end = start + window_size
    window_tokens = tokens[start:end]
    if len(window_tokens) > 0:
      embedding = model.extract_features(window_tokens)
      embedding_list.append(embedding.squeeze(dim = 0).detach().numpy())

  # Combine embeddings from different windows
  if len(embedding_list) > 1:
    combined_embedding = np.concatenate(embedding_list, axis = 0)
  else:
    combined_embedding = embedding_list[0]

  # Save to .pt file
  savefile = savefile + '.pt'
  torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': combined_embedding}, savefile)
  # print(f'Saved to {savefile}')


def get_feature(model, feature_save_root, data_root):
  save = feature_save_root + '/feature/Text/RoBERTa/'
  if not os.path.exists(save):
    os.makedirs(save, exist_ok = True)
  index = 0
  root_directory = data_root
  for root, dirs, files in os.walk(root_directory):
    for name in files:
      if name.endswith('.txt'):
        # txt文件完整路径
        txtfile = os.path.join(root, name)

        parts = name.rsplit('_', 2)
        segment_id = '_'.join(parts[:-1])
        subject_id = '_'.join(parts[:-2])
        # print(f"segment_id:{segment_id} subject_id:{subject_id}")

        # 特征保存路径
        savefile = os.path.join(save, segment_id)
        extract_roberta(model, txtfile, savefile, segment_id, subject_id)
        index += 1
        print(f'第{index}个片段特征提取已完成！')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  feature_save_root = 'data'
  roberta_model_path = 'pretrained_models/roberta.large'
  data_root = 'data/dataset/resample_segments'
  # Load RoBERTa model
  model = RobertaModel.from_pretrained(roberta_model_path, checkpoint_file = 'model.pt')
  model.eval()

  get_feature(model, feature_save_root, data_root)

  print("All Finished!!!")
