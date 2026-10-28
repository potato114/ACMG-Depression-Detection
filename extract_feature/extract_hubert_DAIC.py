import os
import torch
import fairseq
import numpy as np
import soundfile as sf
import scipy.signal as signal
import torch.nn.functional as F
from typing import Tuple, Optional
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)


class Hubert(object):
    def __init__(self, ckpt_path, max_chunk=1600000, wav_length=104640, num_augmented_samples=1):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.max_chunk = max_chunk
        self.wav_length = wav_length
        self.n_samples = 16000
        self.num_augmented_samples = num_augmented_samples

        # ---------- 增加增强功能 ----------
        self._setup_augmentation()  # 初始化增强 pipeline

    # ---------- 新增：增强 pipeline ----------
    def _setup_augmentation(self, config: Optional[dict] = None):
        if config is None:
            config = {
                'noise_prob': 0.5,
                'gain_prob': 0.5,
                'filter_prob': 0.5,
                'delay_prob': 0.5,
                'pitch_shift_prob': 0.5,
                'reverb_prob': 0.5,
            }

        self.train_transform = ComposeMany([
            RandomApply([Noise()], p=config['noise_prob']),
            RandomApply([Gain()], p=config['gain_prob']),
            RandomApply([HighLowPass(sample_rate=16000)], p=config['filter_prob']),
            RandomApply([Delay(sample_rate=16000)], p=config['delay_prob']),
            RandomApply([PitchShift(n_samples=self.n_samples, sample_rate=16000)],
                       p=config['pitch_shift_prob']),
            RandomApply([Reverb(sample_rate=16000)], p=config['reverb_prob']),
        ], num_augmented_samples=self.num_augmented_samples)

    # ---------- 修改 read_audio 增加 augment 参数 ----------
    def read_audio(self, path, augment=False):
        wav, sr = sf.read(path)
        if sr != self.task.cfg.sample_rate:
            num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
            wav = signal.resample(wav, num)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        wav = wav.astype(np.float32)

        # 补齐或裁剪到固定长度
        if wav.shape[0] < self.wav_length:
            wav = np.pad(wav, (0, self.wav_length - wav.shape[0]), constant_values=0)
        else:
            wav = wav[:self.wav_length]

        # 应用增强
        if augment:
            # print("增强ing")
            wav = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)
            wav = self.train_transform(wav)[0]                # (T,)
            wav = wav.cpu().numpy()

        return wav

    # ---------- 修改 read_audio_batch 支持 augment ----------
    def read_audio_batch(self, path_batch, augment=False):
        x = []
        for path in path_batch:
            x.append(self.read_audio(path, augment=augment))
        return np.stack(x, axis=0)

    # ---------- 修改 get_feats 支持 augment ----------
    def get_feats(self, path, layer=12, augment=False):
        if isinstance(path, str):
            B = 1
            x = self.read_audio(path, augment=augment)
        else:
            B = len(path)
            x = self.read_audio_batch(path, augment=augment)

        x = torch.from_numpy(x).float().cuda()
        if self.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(B, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start:start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=layer,
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
  save_L12 = feature_save_root + '/feature/Audio/DAIC_WOZ_hubert_large_L12/'
  save_L24 = feature_save_root + '/feature/Audio/DAIC_WOZ_hubert_large_L24/'
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
        stem = os.path.splitext(name)[0]   # normal303_s0001
        subject_id, _ = stem.rsplit('_', 1) # normal303
        segment_id = stem                  # normal303_s0001

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
  feature_save_root = '/mnt/shareEEx/yuhangbin/Data/Depression'
  ckpt_path = "/mnt/shareEEEx/yuhangbin/Model/Pretrain_Model/hubert_large_ll60k.pt"  # hubert_large_ll60k, hubert_base_ls960
  data_root = '/mnt/shareEEEx/yuhangbin/Data/DAIC-WOZ/segments_participant'

  model = Hubert(ckpt_path)

  get_feature(model, feature_save_root, data_root)

  print("All Finished!!!")
