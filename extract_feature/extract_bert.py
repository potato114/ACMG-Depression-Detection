import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

def extract_bert(model, tokenizer, txtfile, savefile, segment_id, subject_id, window_size=512, stride=512):
    with open(txtfile, 'r') as f:
        text = f.read()

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    embedding_list = []

    # Extract embeddings for each window
    for start in range(0, len(tokens), stride):
        end = start + window_size
        window_tokens = tokens[start:end]
        if len(window_tokens) > 0:
            # Convert tokens to input IDs
            input_ids = torch.tensor([window_tokens])
            with torch.no_grad():
                # Get the embeddings for the current window
                outputs = model(input_ids)
                embedding = outputs.last_hidden_state.squeeze().detach().numpy()
            embedding_list.append(embedding)

    # Combine embeddings from different windows
    if len(embedding_list) > 1:
        combined_embedding = np.concatenate(embedding_list, axis=0)
    else:
        combined_embedding = embedding_list[0]

    # Save to .pt file
    savefile = savefile + '.pt'
    torch.save({'segment_id': segment_id, 'subject_id': subject_id, 'feature': combined_embedding}, savefile)
    print(f'Saved to {savefile}')

def get_feature(model, tokenizer, feature_save_root, data_root):
    save = feature_save_root + '/feature/Text/BERT/'
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)
    index = 0
    root_directory = data_root
    for root, dirs, files in os.walk(root_directory):
        for name in files:
            if name.endswith('.txt'):
                txtfile = os.path.join(root, name)
                parts = name.rsplit('_', 2)
                segment_id = '_'.join(parts[:-1])
                subject_id = '_'.join(parts[:-2])
                savefile = os.path.join(save, segment_id)
                extract_bert(model, tokenizer, txtfile, savefile, segment_id, subject_id)
                index += 1
                print(f'第{index}个片段特征提取已完成！')

if __name__ == '__main__':
    # Set CUDA device if available
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # Define paths
    feature_save_root = 'data'
    model_path = 'pretrained_models/bert-large-chinese'
    data_root = 'data/dataset/resample_segments'

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()

    # Get features
    get_feature(model, tokenizer, feature_save_root, data_root)

    print("All Finished!!!")
