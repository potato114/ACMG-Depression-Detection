import os
import torch
import fairseq
from scipy import io
import soundfile as sf
import scipy.signal as signal
import torch.nn.functional as F


class Wav2vec2(object):
  def __init__(self, ckpt_path, max_chunk = 1600000):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    self.model = model[0].eval().cuda()
    self.task = task
    self.max_chunk = max_chunk

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

  # 获取特征
  def get_feats(self, path, layer):
    '''Layer index starts from 0. (e.g. 0-23)
    '''
    x = self.read_audio(path)
    with torch.no_grad():
      x = torch.from_numpy(x).float().cuda()
      if self.task.cfg.normalize:
        x = F.layer_norm(x, x.shape)
      x = x.view(1, -1)

      feat = []
      for start in range(0, x.size(1), self.max_chunk):
        x_chunk = x[:, start: start + self.max_chunk]
        feat_chunk = self.model.extract_features(source = x_chunk, padding_mask = None, mask = False, layer = layer - 1)

        # feat_chunk is a dict. Keys: ['x', 'padding_mask', 'features', 'layer_results']
        # feat_chunk['x']: Output from transformer stack (target layer / last layer). Shape: B, T, 1024
        # feat_chunk['features']: Output from CNN extractor. Shape: B, T, 512
        # feat_chunk['layer_results'][i][0]: Output from i-th layer. Shape: T, B, 1024
        # Note: feat_chunk['x'] == feat_chunk['layer_results'][-1][0].transpose(0, 1)

        feat.append(feat_chunk['x'])

    return torch.cat(feat, 1).squeeze(0)

def extract_w2v2(model, layer, wavfile, savefile, segment_id, subject_id):
  fea = model.get_feats(wavfile, layer=layer)

  fea = fea.cpu().detach().numpy()   # (t, 1024)

  # Save to .pt file
  savefile = savefile + '.pt'
  torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': fea}, savefile)

  # print(f'Saved to {savefile}')


def get_feature(model, feature_save_root, data_root):

  save_L12 = feature_save_root + '/feature/Audio/w2v2_large_L12_chinese/'
  save_L24 = feature_save_root + '/feature/Audio/w2v2_large_L24_chinese/'

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
        # 12层特征保存路径
        savefile_L12 = os.path.join(save_L12, segment_id)
        # 24层特征保存路径
        savefile_L24 = os.path.join(save_L24, segment_id)
        extract_w2v2(model, 12, wavfile, savefile_L12, segment_id, subject_id)
        extract_w2v2(model, 24, wavfile, savefile_L24, segment_id, subject_id)
        index += 1
        print(f'第{index}个片段特征提取已完成！')

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  feature_save_root = 'data'
  ckpt_path = 'pretrained_models/wav2vec2-large/chinese-wav2vec2-large.pt'
  data_root = 'data/dataset/resample_segments'

  model = Wav2vec2(ckpt_path)

  get_feature(model, feature_save_root, data_root)

  print("All Finished!!!")
