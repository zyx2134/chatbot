markdown
# 🤖 ChatBot Trainer – Train your own chatbot locally

A fully local, offline, no‑pretrained‑model required chatbot trainer.  
Use your own question‑answer data to train a custom chatbot.

## 📦 Requirements

**Required:**
```bash
pip install torch
Optional (for drag & drop support):

bash
pip install tkinterdnd2
▶️ Run
bash
python run.py
📁 Prepare your data
Create one or more JSON files (UTF‑8) with the following structure:

json
[
  {"instruction": "Hello", "output": "Hi! How are you today?"},
  {"instruction": "Thank you", "output": "You're welcome! Happy to help."}
]
instruction – user input

output – bot response

Multiple files are merged automatically during training.

🚀 How to use
Start the program – python run.py

Add data – Click “Add files” and select your JSON file(s)

Choose mode

Fast mode → quick training, good for a few hundred dialogues

Quality mode → slower but smarter

Choose device – CPU or GPU (if available)

Click “Start training” – wait for the progress bar

Chat! – A chat window will pop up automatically after training

💡 Notes
The program remembers all questions from your training data → exact matches get a direct answer.

For unseen questions, it generates answers using the trained model + post‑cleaning → no nonsense.

The best model is saved as best_model.pth – reuse it without retraining.

📄 License
MIT License

text

---

You can now replace the existing `README.md` with this content, then push the change:

```bash
git add README.md
git commit -m "Update README in English"
git push
