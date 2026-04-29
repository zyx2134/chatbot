import json
import os
import re
import torch
from collections import defaultdict, Counter

def load_instruction_output_pairs(file_paths):
    """从 JSON 文件加载 instruction/output 对，同时构建检索字典"""
    pairs = []
    ins2outs = defaultdict(list)
    for fpath in file_paths:
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            try:
                if "instruction" in item and "output" in item:
                    q = item["instruction"].strip()
                    a = item["output"].strip()
                elif "conversation" in item:
                    conv = item["conversation"]
                    q = conv[0]["message"].strip()
                    a = conv[1]["message"].strip()
                else:
                    continue
                if q and a:
                    pairs.append((q, a))
                    ins2outs[q].append(a)
            except:
                continue
    return pairs, ins2outs

def clean_conversation(q, a, max_input_len, max_output_len, min_response_len):
    if not q or not a:
        return False
    if len(q) < 2 or len(q) > max_input_len * 2:
        return False
    if len(a) < min_response_len or len(a) > max_output_len * 2:
        return False
    if re.fullmatch(r'[，。！？；：""''、\s\d]+', a):
        return False
    if len(a) > 3 and max(Counter(a).values()) / len(a) > 0.7:
        return False
    return True

def build_char_vocab(pairs):
    all_chars = set()
    for q, a in pairs:
        all_chars.update(q)
        all_chars.update(a)
    specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab = specials + sorted(all_chars)
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char, len(vocab)

def encode(text, char2idx, max_len, add_sos_eos=False):
    ids = []
    if add_sos_eos:
        ids.append(char2idx["<SOS>"])
    for ch in text:
        ids.append(char2idx.get(ch, char2idx["<UNK>"]))
    if add_sos_eos:
        ids.append(char2idx["<EOS>"])
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [char2idx["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode(ids, idx2char):
    chars = []
    for i in ids:
        if i not in (0, 1, 2):
            chars.append(idx2char[i])
    return "".join(chars)