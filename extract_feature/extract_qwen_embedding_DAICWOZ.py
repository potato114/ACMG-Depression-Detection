import os
import torch
import numpy as np
from typing import Optional
import pandas as pd
from modelscope import AutoTokenizer, AutoModel  # model-specific tokenizers are registered here

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  """
  Qwen3-Embedding recommended pooling: use the last *non-pad* token embedding.
  Works for both left- and right-padding.
  """
  left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
  if left_padding:
    return last_hidden_states[:, -1]
  sequence_lengths = attention_mask.sum(dim=1) - 1
  batch_size = last_hidden_states.shape[0]
  return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
  # Qwen3-Embedding recommended prompt format for "query" style inputs.
  return f"Instruct: {task_description}\nQuery:{query}"


def extract_qwen_embedding(
  model,
  tokenizer,
  device: torch.device,
  text: str,
  savefile: str,
  segment_id: str,
  subject_id: str,
  window_size: int = 8192,
  stride: int = 8192,
  pooling: str = "last_token",
  use_instruct: bool = True,
  task_description: str = "Given a text, extract an embedding that focuses on expressions related to depression and depressive symptoms.",
  output_mode: str = "token",
) -> None:
  """
  Qwen3-Embedding feature extraction (recommended usage):
    - tokenize to tensors with attention_mask
    - run model -> last_hidden_state
    - pool embedding via last_token_pool (default) and L2-normalize
    - for long text, use token-id sliding window (concat pooled embeddings per window)
      so the saved tensor shape is [num_windows, D]
    - optionally prepend an instruction prompt (Qwen3-Embedding query format)

  output_mode:
    - "token": save per-token embeddings (aligned with extract_roberta.py) -> [N, D]
    - "pooled": save one embedding per window -> [num_windows, D]
  """
  if not isinstance(text, str) or text.strip() == "":
    return

  pooling = str(pooling).strip().lower()
  output_mode = str(output_mode).strip().lower()

  # Instruction prefix tokens (Qwen3-Embedding query-style prompt).
  prefix_token_ids = []
  if use_instruct and isinstance(task_description, str) and task_description.strip():
    prefix_token_ids = tokenizer.encode(f"Instruct: {task_description}\nQuery:", add_special_tokens=False)
  prefix_len = len(prefix_token_ids)

  # We keep a token-level sliding window, but Qwen3-Embedding supports large max_length (e.g. 8192).
  token_ids = tokenizer.encode(text, add_special_tokens=False)
  if len(token_ids) == 0:
    return

  content_window_size = int(window_size) - prefix_len
  if content_window_size <= 0:
    raise ValueError(
      f"window_size={window_size} is too small for instruction prefix (len={prefix_len}). "
      "Increase window_size or disable use_instruct."
    )

  embedding_list = []
  num_special = tokenizer.num_special_tokens_to_add(pair=False)

  for start in range(0, len(token_ids), int(stride)):
    end = start + content_window_size
    window_token_ids = token_ids[start:end]
    if not window_token_ids:
      continue

    # Build token ids exactly (for consistent slicing and to be able to drop prefix embeddings).
    input_ids = tokenizer.build_inputs_with_special_tokens(prefix_token_ids + window_token_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    with torch.no_grad():
      outputs = model(**batch)

    if output_mode == "pooled":
      if pooling == "last_token":
        emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])  # [1,D]
      elif pooling == "cls":
        emb = outputs.last_hidden_state[:, 0]  # [1,D]
      else:
        raise ValueError(f"Unsupported pooling: {pooling}. Use 'last_token' or 'cls'.")
      emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # [1,D]
      embedding_list.append(emb.squeeze(0).detach().cpu().to(torch.float32).numpy())
    elif output_mode == "token":
      hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32).numpy()  # [T,D]
      # Drop special tokens like <bos>/<eos> to match extract_roberta.py behavior.
      if num_special >= 2 and hidden.shape[0] > num_special:
        hidden = hidden[1:-1]
      # If we used an instruction prefix, drop its embeddings so N tracks the text content tokens.
      if prefix_len and hidden.shape[0] > prefix_len:
        hidden = hidden[prefix_len : prefix_len + len(window_token_ids)]
      embedding_list.append(hidden)
    else:
      raise ValueError(f"Unsupported output_mode: {output_mode}. Use 'token' or 'pooled'.")

  if not embedding_list:
    return

  if output_mode == "pooled":
    combined_embedding = np.stack(embedding_list, axis=0)  # [num_windows, D]
  else:
    combined_embedding = np.concatenate(embedding_list, axis=0) if len(embedding_list) > 1 else embedding_list[0]  # [N, D]

  savefile = savefile + ".pt"
  feature_tensor = torch.from_numpy(combined_embedding).to(torch.float32)
  torch.save({"segment_id": segment_id, "subject_id": subject_id, "feature": feature_tensor}, savefile)


def get_feature_from_excel(
  model,
  tokenizer,
  device: torch.device,
  feature_save_root: str,
  excel_path: str,
  text_column: str = "模型情感描述",
  segment_id_column: str = "文件名",
  save_subdir: str = "feature/Text_d/emotion_Qwen3Embedding/",
  use_instruct: bool = True,
  task_description: str = "Given a text, extract an embedding that focuses on expressions related to depression and depressive symptoms.",
  output_mode: str = "token",
) -> None:
  save = os.path.join(feature_save_root, save_subdir)
  os.makedirs(save, exist_ok=True)

  if excel_path.endswith(".xlsx") or excel_path.endswith(".xls"):
    df = pd.read_excel(excel_path)
  else:
    df = pd.read_csv(excel_path)

  index = 0
  for _, row in df.iterrows():
    segment_id = row.get(segment_id_column)
    text = row.get(text_column)
    if not isinstance(text, str) or not isinstance(segment_id, str):
      continue

    name_no_ext = os.path.splitext(segment_id)[0]
    # Keep the same segment_id parsing convention as extract_roberta.py.
    segment_id = "_".join(name_no_ext.split("_")[:-1])
    if not segment_id:
      continue

    subject_id = segment_id.split("_")[0]
    savefile = os.path.join(save, segment_id)
    extract_qwen_embedding(
      model,
      tokenizer,
      device,
      text,
      savefile,
      segment_id,
      subject_id,
      use_instruct=use_instruct,
      task_description=task_description,
      output_mode=output_mode,
    )

    index += 1
    print(f"第{index}个片段特征提取已完成！")


def get_feature_from_txt_dir(
  model,
  tokenizer,
  device: torch.device,
  feature_save_root: str,
  data_root: str,
  save_subdir: str = "feature/Text/Qwen3Embedding-DAIC-WOZ/",
  use_instruct: bool = True,
  task_description: str = "Represent the speaker’s language content and emotional cues without making any diagnosis or judgment.",
  output_mode: str = "token",
) -> None:
  """
  Match the older fairseq-roberta extraction workflow:
    - traverse a directory tree looking for *.txt
    - parse segment_id/subject_id from filename like:
        parts = name.rsplit('_', 2)
        segment_id = '_'.join(parts[:-1])
        subject_id = '_'.join(parts[:-2])
    - read the whole txt and extract features with sliding window
  """
  save = os.path.join(feature_save_root, save_subdir)
  os.makedirs(save, exist_ok=True)

  index = 0
  for root, _, files in os.walk(data_root):
    for name in files:
      if not name.endswith(".txt"):
        continue
      txtfile = os.path.join(root, name)
      try:
        with open(txtfile, "r", encoding="utf-8") as f:
          text = f.read()
      except Exception:
        # Fallback for non-utf8 files.
        with open(txtfile, "r", errors="ignore") as f:
          text = f.read()

      stem = os.path.splitext(name)[0]   # normal303_s0001
      subject_id, _ = stem.rsplit('_', 1) # normal303
      segment_id = stem                  # normal303_s0001

      if not segment_id or not subject_id:
        continue

      savefile = os.path.join(save, segment_id)
      extract_qwen_embedding(
        model,
        tokenizer,
        device,
        text,
        savefile,
        segment_id,
        subject_id,
        use_instruct=use_instruct,
        task_description=task_description,
        output_mode=output_mode,
      )
      index += 1
      print(f"第{index}个片段特征提取已完成！")


def load_qwen_embedding(model_path: str, device: torch.device, torch_dtype: Optional[str] = None):
  """
  Load local Qwen3-Embedding model.
  Prefer ModelScope's AutoTokenizer/AutoModel for Qwen3-Embedding.
  """
  dtype = None
  if torch_dtype:
    s = str(torch_dtype).lower().strip()
    if s in ("fp16", "float16"):
      dtype = torch.float16
    elif s in ("bf16", "bfloat16"):
      dtype = torch.bfloat16
    elif s in ("fp32", "float32"):
      dtype = torch.float32



  tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
  model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
  model.to(device)
  model.eval()
  return model, tokenizer


if __name__ == "__main__":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

  feature_save_root = "/mnt/shareEEx/yuhangbin/Data/Depression"
  qwen_model_path = "/mnt/shareEEx/yuhangbin/model/Qwen3-Embedding-0.6B"
  data_root = "/mnt/shareEEEx/yuhangbin/Data/DAIC-WOZ/segments_participant"
  task = "Given a text, extract an embedding that focuses on emotional expressions and psychological signals related to depression, without making any diagnosis."

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, tokenizer = load_qwen_embedding(qwen_model_path, device=device, torch_dtype="bf16" if device.type == "cuda" else "fp32")

  get_feature_from_txt_dir(model, tokenizer, device, feature_save_root, data_root, use_instruct=True, task_description=task)
  print("All Finished!!!")
