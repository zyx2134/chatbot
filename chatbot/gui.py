import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
from .config import FastConfig, QualityConfig
from .train import train_model
from .inference import load_model_from_checkpoint, generate_response
import torch

# 尝试导入拖拽
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

def open_chat_window():
    # 加载最佳模型
    if os.path.exists(QualityConfig.SAVE_PATH):
        path = QualityConfig.SAVE_PATH
    elif os.path.exists(FastConfig.SAVE_PATH):
        path = FastConfig.SAVE_PATH
    else:
        messagebox.showerror("错误", "未找到模型文件，请先训练")
        return
    model, char2idx, idx2char, _, ins2outs, _, mode = load_model_from_checkpoint(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    win = tk.Toplevel()
    win.title(f"对话机器人 - {'快速' if mode=='fast' else '高质量'}模式")
    win.geometry("550x650")
    win.configure(bg='#2b2b2b')
    chat_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("微软雅黑", 11), bg='#1e1e1e', fg='white')
    chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    chat_area.insert(tk.END, "机器人: 你好！直接问我问题吧。\n")
    chat_area.config(state=tk.DISABLED)

    frame = tk.Frame(win, bg='#2b2b2b')
    frame.pack(fill=tk.X, padx=10, pady=5)
    entry = tk.Entry(frame, font=("微软雅黑", 11), bg='#3c3f41', fg='white')
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    def send():
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, tk.END)
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"你: {msg}\n")
        chat_area.see(tk.END)
        chat_area.config(state=tk.DISABLED)
        def gen():
            resp = generate_response(msg, model, char2idx, idx2char, device, ins2outs)
            chat_area.config(state=tk.NORMAL)
            chat_area.insert(tk.END, f"机器人: {resp}\n\n")
            chat_area.see(tk.END)
            chat_area.config(state=tk.DISABLED)
        threading.Thread(target=gen, daemon=True).start()
    send_btn = tk.Button(frame, text="发送", command=send, bg='#5cb85c', fg='white')
    send_btn.pack(side=tk.RIGHT, padx=5)
    entry.bind("<Return>", lambda e: send())

class App:
    def __init__(self, root):
        self.root = root
        root.title("对话训练器 - 拖拽或按钮添加文件")
        root.geometry("680x600")
        root.configure(bg='#2b2b2b')
        self.files = []

        # 拖拽/按钮区
        top_frame = tk.Frame(root, bg='#3c3f41', height=180)
        top_frame.pack(fill=tk.X, padx=20, pady=15)
        top_frame.pack_propagate(False)
        lbl = tk.Label(top_frame, text="拖拽 JSON 文件到此处（可多个）\n或者点击下方按钮选择文件",
                       bg='#3c3f41', fg='white', font=("微软雅黑", 11))
        lbl.pack(expand=True)

        if HAS_DND:
            top_frame.drop_target_register(DND_FILES)
            top_frame.dnd_bind('<<Drop>>', self.on_drop)
        else:
            tk.Label(top_frame, text="拖拽不可用（需安装 tkinterdnd2）", bg='#3c3f41', fg='yellow').pack()

        btn_frame = tk.Frame(top_frame, bg='#3c3f41')
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="添加文件", command=self.add_files, bg='#4cae4c').pack()

        # 文件列表
        list_frame = tk.Frame(root, bg='#2b2b2b')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        tk.Label(list_frame, text="已添加的文件：", bg='#2b2b2b', fg='white', anchor='w').pack(fill=tk.X)
        self.listbox = tk.Listbox(list_frame, bg='#4a4e51', fg='white')
        self.listbox.pack(fill=tk.BOTH, expand=True)
        btn2_frame = tk.Frame(list_frame, bg='#2b2b2b')
        btn2_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn2_frame, text="移除选中", command=self.remove_selected, bg='#e66a5c', fg='white').pack(side=tk.LEFT)

        # 模式与设备
        opt_frame = tk.Frame(root, bg='#2b2b2b')
        opt_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(opt_frame, text="训练模式:", bg='#2b2b2b', fg='white').grid(row=0, column=0, sticky='w')
        self.mode_var = tk.StringVar(value='fast')
        tk.Radiobutton(opt_frame, text="快速模式", variable=self.mode_var, value='fast',
                       bg='#2b2b2b', fg='white', selectcolor='#2b2b2b').grid(row=0, column=1, padx=5)
        tk.Radiobutton(opt_frame, text="高质量模式", variable=self.mode_var, value='quality',
                       bg='#2b2b2b', fg='white', selectcolor='#2b2b2b').grid(row=0, column=2, padx=5)

        tk.Label(opt_frame, text="训练设备:", bg='#2b2b2b', fg='white').grid(row=1, column=0, sticky='w', pady=5)
        self.device_var = tk.StringVar(value='cpu')
        tk.Radiobutton(opt_frame, text="CPU", variable=self.device_var, value='cpu',
                       bg='#2b2b2b', fg='white', selectcolor='#2b2b2b').grid(row=1, column=1, padx=5)
        if torch.cuda.is_available():
            tk.Radiobutton(opt_frame, text="GPU (CUDA)", variable=self.device_var, value='cuda',
                           bg='#2b2b2b', fg='white', selectcolor='#2b2b2b').grid(row=1, column=2, padx=5)

        # 进度条和状态
        self.progress = ttk.Progressbar(root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        self.status = tk.Label(root, text="就绪", bg='#2b2b2b', fg='#aaaaaa')
        self.status.pack(fill=tk.X, padx=20)

        self.train_btn = tk.Button(root, text="开始训练", command=self.start_training,
                                   bg='#5cb85c', fg='white', font=("微软雅黑", 12, "bold"))
        self.train_btn.pack(pady=15)

    def on_drop(self, event):
        raw = event.data.strip('{}')
        parts = raw.split()
        for p in parts:
            p = os.path.normpath(p)
            if p.endswith('.json') and p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, os.path.basename(p))
        self.status.config(text=f"已添加 {len(self.files)} 个文件")

    def add_files(self):
        new = filedialog.askopenfilenames(filetypes=[("JSON", "*.json")])
        for f in new:
            if f not in self.files:
                self.files.append(f)
                self.listbox.insert(tk.END, os.path.basename(f))
        self.status.config(text=f"已添加 {len(self.files)} 个文件")

    def remove_selected(self):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            del self.files[idx]
            self.listbox.delete(idx)

    def start_training(self):
        if not self.files:
            messagebox.showwarning("警告", "请先添加JSON文件")
            return
        self.train_btn.config(state=tk.DISABLED, text="训练中...")
        self.progress['value'] = 0
        self.status.config(text="开始训练...")
        mode = self.mode_var.get()
        device = self.device_var.get()
        def update_progress(ep, total, loss):
            self.progress['value'] = (ep / total) * 100
            self.root.update_idletasks()
        def update_status(msg):
            self.status.config(text=msg)
            self.root.update_idletasks()
        def train():
            ok = train_model(self.files, mode, device, update_progress, update_status)
            self.root.after(0, lambda: self.training_done(ok))
        threading.Thread(target=train, daemon=True).start()

    def training_done(self, success):
        self.train_btn.config(state=tk.NORMAL, text="开始训练")
        if success:
            messagebox.showinfo("完成", "训练成功！即将打开聊天窗口。")
            open_chat_window()
        else:
            messagebox.showerror("失败", "训练出错，请查看控制台输出")