import io
import os
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# 配置（默认值，可在函数调用时覆盖）
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


# 简单分词：按空格为单位（若需要子词/更专业的 tokenization，可替换）
def tokenize(text: str) -> List[str]:
    return text.strip().split()


def build_vocab(lines: List[str], vocab_size: int):
    counter = Counter()
    for line in lines:
        for w in tokenize(line):
            counter[w] += 1
    most_common = counter.most_common(vocab_size - 4)  # 保留 4 个 special tokens
    idx2tok = [pad_token, unk_token, bos_token, eos_token] + [w for w, _ in most_common]
    tok2idx = {tok: idx for idx, tok in enumerate(idx2tok)}
    return tok2idx, idx2tok


def text_to_ids(text: str, tok2idx: dict) -> List[int]:
    tokens = tokenize(text)
    ids = [tok2idx.get(t, UNK_IDX) for t in tokens]
    return ids


def build_dataset(lines: List[str], tok2idx: dict, max_seq_length: int) -> List[Tuple[List[int], List[int]]]:
    """将文本行转换为 (src, tgt) 对，均为定长 max_seq_length 的 id 列表

    将每行 tokens 按 window = max_seq_length + 1 切分，
    对每个 chunk 生成 src=chunk[:-1], tgt=chunk[1:]。如果 chunk 不足长度，
    使用 PAD 填充。保证 src 与 tgt 长度一致且不会越界。
    """
    ids_list = [text_to_ids(line, tok2idx) for line in lines]
    samples = []
    window = max_seq_length + 1
    for ids in ids_list:
        if len(ids) < 2:
            continue
        for i in range(0, len(ids), window):
            chunk = ids[i : i + window]
            if len(chunk) < 2:
                continue
            # src: first L, tgt: last L
            if len(chunk) < window:
                # pad chunk to window
                pad_len = window - len(chunk)
                chunk = chunk + [PAD_IDX] * pad_len
            src = chunk[:-1]
            tgt = chunk[1:]
            samples.append((src, tgt))
    return samples


def read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


class SequenceDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[int], List[int]]]):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    # batch 是 list of (src_tensor, tgt_tensor)，它们已经是定长的
    srcs = torch.stack([item[0] for item in batch], dim=0)
    tgts = torch.stack([item[1] for item in batch], dim=0)
    return srcs, tgts


def find_wikitext_files(data_dir: str):
    # 尝试常见文件名
    candidates = {
        'train': ['wiki.train.tokens', 'train.txt', 'train.tokens', 'wiki.train'],
        'valid': ['wiki.valid.tokens', 'valid.txt', 'valid.tokens', 'wiki.valid'],
        'test':  ['wiki.test.tokens', 'test.txt', 'test.tokens', 'wiki.test']
    }
    paths = {}
    for split, names in candidates.items():
        found = None
        for n in names:
            p = os.path.join(data_dir, n)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Cannot find {split} file in {data_dir}; tried: {names}")
        paths[split] = found
    return paths


def get_dataloaders(data_dir: str,
                    batch_size: int = 32,
                    vocab_size: int = 10000,
                    max_seq_length: int = 100,
                    shuffle: bool = True,
                    num_workers: int = 0):
    """构建并返回 tok2idx, idx2tok, train_loader, valid_loader, test_loader, pad_idx

    返回值方便训练脚本直接使用：
      tok2idx, idx2tok, train_loader, valid_loader, test_loader, pad_idx
    """
    files = find_wikitext_files(data_dir)
    train_lines = read_lines(files['train'])
    valid_lines = read_lines(files['valid'])
    test_lines  = read_lines(files['test'])

    all_lines = train_lines + valid_lines + test_lines
    tok2idx, idx2tok = build_vocab(all_lines, vocab_size)

    train_samples = build_dataset(train_lines, tok2idx, max_seq_length)
    valid_samples = build_dataset(valid_lines, tok2idx, max_seq_length)
    test_samples  = build_dataset(test_lines, tok2idx, max_seq_length)

    train_dataset = SequenceDataset(train_samples)
    valid_dataset = SequenceDataset(valid_samples)
    test_dataset  = SequenceDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return tok2idx, idx2tok, train_loader, valid_loader, test_loader, PAD_IDX


if __name__ == "__main__":
    # 简单本地测试（仅在直接运行脚本时）
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'wikitext-2')
    data_dir = os.path.abspath(data_dir)
    print('Looking for dataset in', data_dir)
    tok2idx, idx2tok, train_loader, valid_loader, test_loader, pad_idx = get_dataloaders(
        data_dir, batch_size=8, vocab_size=5000, max_seq_length=32)
    batch = next(iter(train_loader))
    print('Sample batch shapes:', batch[0].shape, batch[1].shape)