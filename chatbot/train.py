import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .config import FastConfig, QualityConfig
from .model import Seq2Seq
from .data_utils import load_instruction_output_pairs, clean_conversation, build_char_vocab, encode

class ChatDataset(Dataset):
    def __init__(self, pairs, char2idx, max_input_len, max_output_len):
        self.X = [encode(q, char2idx, max_input_len, False) for q,_ in pairs]
        self.Y = [encode(a, char2idx, max_output_len, True) for _,a in pairs]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(files, mode, device_str, progress_cb, status_cb):
    try:
        # 加载数据
        status_cb("加载数据...")
        all_pairs, ins2outs = load_instruction_output_pairs(files)
        if not all_pairs:
            status_cb("未找到有效对话数据")
            return False
        status_cb(f"加载 {len(all_pairs)} 条对话")

        if mode == 'fast':
            cfg = FastConfig()
        else:
            cfg = QualityConfig()

        # 清洗
        cleaned_pairs = []
        for q, a in all_pairs:
            if clean_conversation(q, a, cfg.MAX_INPUT_LEN, cfg.MAX_OUTPUT_LEN, cfg.MIN_RESPONSE_LEN):
                cleaned_pairs.append((q, a))
        status_cb(f"清洗后 {len(cleaned_pairs)} 条对话")

        # 词汇表
        char2idx, idx2char, vocab_size = build_char_vocab(cleaned_pairs)
        status_cb(f"词汇表大小: {vocab_size}")

        # 数据集
        dataset = ChatDataset(cleaned_pairs, char2idx, cfg.MAX_INPUT_LEN, cfg.MAX_OUTPUT_LEN)
        loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

        # 设备
        device = torch.device(device_str if torch.cuda.is_available() and device_str=='cuda' else 'cpu')
        status_cb(f"设备: {device}")

        # 模型
        model = Seq2Seq(vocab_size, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        status_cb(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练
        model.train()
        best_loss = float('inf')
        for epoch in range(cfg.EPOCHS):
            total_loss = 0
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = model(src, tgt, teacher_ratio=cfg.TEACHER_FORCING_RATIO)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            progress_cb(epoch+1, cfg.EPOCHS, avg_loss)
            if (epoch+1) % 10 == 0:
                status_cb(f"Epoch {epoch+1}/{cfg.EPOCHS} Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'mode': mode,
                    'model_state': model.state_dict(),
                    'char2idx': char2idx,
                    'idx2char': idx2char,
                    'vocab_size': vocab_size,
                    'config': cfg.__dict__,
                    'ins2outs': dict(ins2outs)
                }, cfg.SAVE_PATH)
        status_cb(f"训练完成！最佳 Loss: {best_loss:.4f}")
        return True
    except Exception as e:
        status_cb(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        return False