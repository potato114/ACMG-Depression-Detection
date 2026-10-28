import os
import torch
import fairseq
import numpy as np
import soundfile as sf
import scipy.signal as signal
import torch.nn.functional as F


class Hubert(object):
  def __init__(self, ckpt_path, max_chunk = 1600000, wav_length = 104640):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    self.model = model[0].eval().cuda()
    self.task = task
    self.max_chunk = max_chunk
    self.wav_length = wav_length  # (326 + 1) * 0.02 * 16000 = 104640

  # 读取音频数据
  def read_audio(self, path):
    # 读取音频数据和采样率
    wav, sr = sf.read(path)
    # 若音频不符合指定采样率，重采样为指定采样率
    if sr != self.task.cfg.sample_rate:
      num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
      wav = signal.resample(wav, num)
      print(f'Resample {sr} to {self.task.cfg.sample_rate}')
    # 指定wav为一维数据
    if wav.ndim == 2:
      wav = wav.mean(-1)
    assert wav.ndim == 1, wav.ndim
    # 返回wav音频数据
    return wav

  # 批量读取音频数据
  def read_audio_batch(self, path_batch):
    x = []
    for path in path_batch:
      _x = self.read_audio(path)
      _x = np.pad(_x, (0, self.wav_length - _x.shape[0]), constant_values = (0, 0) ) if _x.shape[0] < self.wav_length else _x[:self.wav_length]
      x.append(_x)
    x = np.stack(x, axis = 0)
    return x

  # 获取特征
  def get_feats(self, path, layer):
    '''Layer index starts from 1. (e.g. 1-24)
    '''
    if isinstance(path, str):
      B = 1
      x = self.read_audio(path)
    else:
      B = len(path)
      x = self.read_audio_batch(path)

    x = torch.from_numpy(x).float().cuda()
    if self.task.cfg.normalize:
      x = F.layer_norm(x, x.shape)
    x = x.view(B, -1)

    feat = []
    for start in range(0, x.size(1), self.max_chunk):
      x_chunk = x[:, start: start + self.max_chunk]
      feat_chunk, _ = self.model.extract_features(
        source = x_chunk,
        padding_mask = None,
        mask = False,
        output_layer = layer,
      )
      feat.append(feat_chunk)

    return torch.cat(feat, 1)


def extract_hubert(model, layer, wavfile, savefile, segment_id, subject_id):
  with torch.no_grad():
    fea = model.get_feats(wavfile, layer = layer).squeeze(0)

  fea = fea.cpu().detach().numpy()

  # Save to .pt file
  savefile = savefile + '.pt'
  torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': fea}, savefile)
  # print(f'segment_id:{segment_id}  subject_id:{subject_id}')
  # print(f'Saved to {savefile}')


def get_feature(model, feature_save_root, data_root):
  save_L12 = feature_save_root + '/feature/Audio/hubert_large_L12_chinese/'
  save_L24 = feature_save_root + '/feature/Audio/hubert_large_L24_chinese/'
  if not os.path.exists(save_L12):
    os.makedirs(save_L12, exist_ok= True)
  if not os.path.exists(save_L24):
    os.makedirs(save_L24, exist_ok= True)
  index = 0
  root_directory = data_root
  for root, dirs, files in os.walk(root_directory):
    for name in files:
      if name.endswith('.wav'):
        # wav文件完整路径
        wavfile = os.path.join(root, name)
        parts = name.rsplit('_', 2)
        segment_id = '_'.join(parts[:-1])
        subject_id = '_'.join(parts[:-2])
        # print(f"segment_id:{segment_id} subject_id:{subject_id}")
        # 12层特征保存路径
        savefile_L12 = os.path.join(save_L12, segment_id)
        # 24层特征保存路径
        savefile_L24 = os.path.join(save_L24, segment_id)
        extract_hubert(model, 12, wavfile, savefile_L12, segment_id, subject_id)
        extract_hubert(model, 24, wavfile, savefile_L24, segment_id, subject_id)
        index += 1
        print(f'第{index}个片段特征提取已完成！')

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  feature_save_root = 'data'
  ckpt_path = 'pretrained_models/hubert-large/chinese-hubert-large.pt'  # hubert_large_ll60k, hubert_base_ls960
  data_root = 'data/dataset/resample_segments'
  model = Hubert(ckpt_path)

  get_feature(model, feature_save_root, data_root)

  print("All Finished!!!")
