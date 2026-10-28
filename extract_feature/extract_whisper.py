import torch
import whisper
import os

def extract_features(model, wavfile, savefile, segment_id, subject_id):
    # 加载音频
    audio = whisper.load_audio(wavfile)

    # 记录音频的原始长度
    original_length = audio.shape[0]

    # 重新采样到 16kHz 并预处理
    audio = whisper.pad_or_trim(audio)

    # 预处理音频并移动到指定的 GPU
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)

    # 获取模型编码器输出
    with torch.no_grad():
        # 提取编码器的特征
        encoder_output = model.encoder(mel)

    # 特征
    fea = encoder_output.squeeze().cpu().detach().numpy()

    # 计算填充长度
    mel_length = mel.shape[-1]
    padding_length = mel_length - original_length

    # 去除填充部分
    if padding_length > 0:
        fea = fea[:, :-padding_length]  # 去除末尾的填充部分

    # Save to .pt file
    savefile = savefile + '.pt'
    torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': fea}, savefile)
    print(f'Saved to {savefile}')

    # 清理内存
    del audio, mel, encoder_output
    torch.cuda.empty_cache()

def get_feature(model, feature_save_root, data_root):
    save = feature_save_root + '/feature/Audio/whisper/'
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
    # 指定使用 GPU 3
    device = torch.device("cuda:3")

    # Define paths
    feature_save_root = 'data'
    data_root = 'data/dataset/resample_segments'

    # 加载 Whisper 模型并移动到指定的 GPU
    model = whisper.load_model("small").to(device)

    # Get features
    get_feature(model, feature_save_root, data_root)

    print("All Finished!!!")
