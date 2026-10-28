import os
import torch
import torchaudio
from WavLM import WavLM, WavLMConfig

# 指定使用的GPU设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def extract_features(model, wavfile, savefile, segment_id, subject_id):
    waveform, sample_rate = torchaudio.load(wavfile)

    # 如果采样率不是16kHz，则需要重采样
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # 将waveform移动到GPU
    waveform = waveform.to(device)

    # 如果模型需要对输入归一化
    if cfg.normalize:
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

    # 将音频输入到模型中以提取特征
    with torch.no_grad():
        rep = model.extract_features(waveform)[0]

    # 去掉第一个维度（batch维度）
    fea = torch.squeeze(rep, dim=0)

    # 将特征移回CPU并保存为.pt文件
    fea = fea.cpu().detach().numpy()
    savefile = savefile + '.pt'
    torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': fea}, savefile)
    print(f'Saved to {savefile}')


def get_feature(model, feature_save_root, data_root):
    save = feature_save_root + '/feature/Audio/wavlm/'
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)

    index = 0
    root_directory = data_root
    for root, dirs, files in os.walk(root_directory):
        for name in files:
            if name.endswith('.wav'):
                wavfile = os.path.join(root, name)
                parts = name.rsplit('_', 2)
                segment_id = '_'.join(parts[:-1])
                subject_id = '_'.join(parts[:-2])
                savefile = os.path.join(save, segment_id)
                extract_features(model, wavfile, savefile, segment_id, subject_id)
                index += 1
                print(f'第{index}个片段特征提取已完成！')


if __name__ == '__main__':

    # Define paths
    feature_save_root = 'data'
    data_root = 'data/dataset/resample_segments'

    # 加载预训练的模型检查点
    checkpoint_path = 'pretrained_models/WavLM-Base.pt'
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 将模型移动到指定的GPU
    model = model.to(device)

    # Get features
    get_feature(model, feature_save_root, data_root)

    print("All Finished!!!")
