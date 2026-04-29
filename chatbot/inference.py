import re
import random
import torch
from .model import Seq2Seq
from .data_utils import decode

def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    mode = checkpoint.get('mode', 'fast')
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    vocab_size = checkpoint['vocab_size']
    cfg = checkpoint['config']
    ins2outs = checkpoint.get('ins2outs', {})
    embed_dim = cfg.get('EMBED_DIM', 64)
    hidden_dim = cfg.get('HIDDEN_DIM', 128)
    num_layers = cfg.get('NUM_LAYERS', 1)
    model = Seq2Seq(vocab_size, embed_dim, hidden_dim, num_layers)
    model.load_state_dict(checkpoint['model_state'])
    return model, char2idx, idx2char, vocab_size, ins2outs, cfg, mode

def generate_response(text, model, char2idx, idx2char, device, ins2outs, max_len=60):
    # 检索优先
    if text in ins2outs:
        return random.choice(ins2outs[text])

    model.eval()
    ids = [char2idx.get(ch, char2idx["<UNK>"]) for ch in text]
    max_input = 40
    if len(ids) > max_input:
        ids = ids[:max_input]
    else:
        ids += [0] * (max_input - len(ids))
    src = torch.tensor([ids], device=device)
    with torch.no_grad():
        hidden, cell = model.enc(src)
        dec_input = torch.tensor([char2idx["<SOS>"]], device=device)
        result_ids = []
        for _ in range(max_len):
            logits, hidden, cell = model.dec(dec_input, hidden, cell)
            top = logits.argmax(1).item()
            if top == char2idx["<EOS>"]:
                break
            if top != char2idx["<PAD>"]:
                result_ids.append(top)
            dec_input = torch.tensor([top], device=device)
        raw = decode(result_ids, idx2char)
        raw = re.sub(r'([，。！？；：""''、])\1+', r'\1', raw)
        raw = raw.strip('，。！？；：""''、')
        if len(raw) < 2:
            all_outs = [out for outs in ins2outs.values() for out in outs]
            if all_outs:
                raw = random.choice(all_outs)
            else:
                raw = "你好，有什么事吗？"
        return raw