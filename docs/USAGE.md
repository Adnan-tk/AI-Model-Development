## Usage Guide

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Run the UI
```bash
streamlit run app.py
```

Workflow in the app:
- Upload CSV/Excel/JSON/TXT files in Data Upload
- Process data to clean text and train a tokenizer
- Configure model and start training
- Chat with the trained model in Chat

### 3) Programmatic Data Prep and Model
```python
from data_processing.data_loader import DataLoader
from data_processing.preprocessor import TextPreprocessor
from model.transformer import DeepSeekChatbot
from model.trainer import ChatbotTrainer
import torch

# Load data
dl = DataLoader()
df = dl.load_csv("sample_data/conversations.csv")
pairs = dl.get_conversation_pairs("question", "answer")

# Prepare texts and tokenizer
tp = TextPreprocessor()
texts = [f"{q} {a}" for q, a in pairs]
sp = tp.train_tokenizer(texts, vocab_size=8000)

# Tokenize for a tiny example
inputs = torch.tensor(tp.tokenize([q for q,_ in pairs][:32], max_length=64))
targets = torch.tensor(tp.tokenize([a for _,a in pairs][:32], max_length=64))

# Model and training
model = DeepSeekChatbot(vocab_size=tp.vocab_size, d_model=256, num_heads=4, num_layers=4)
trainer = ChatbotTrainer(model, vocab_size=tp.vocab_size)
# trainer.train(train_loader, val_loader)  # Prepare loaders as needed

# Inference (IDs -> decode using SentencePiece)
prompt_ids = torch.tensor([sp.encode("hello")]).long()
gen_ids = model.generate(prompt_ids, max_length=30)
print(sp.decode(gen_ids[0].tolist()))
```

### 4) Hugging Face Trainer Pipeline
```python
from train_chatbot import ChatbotTrainer
hf = ChatbotTrainer()
convs = hf.load_data_from_csv("sample_data/conversations.csv")
tok = hf.tokenize_data(convs)
hf.train(tok, epochs=1)
print(hf.generate_response("Hi there"))
```

### Troubleshooting
- Ensure special tokens are configured once (pad/bos/eos). The tokenizer trainer avoids duplicates.
- For GPU training, install CUDA-enabled PyTorch and ensure `torch.cuda.is_available()` is true.
- PDFs and DOCX require `PyPDF2` and `python-docx` respectively.

