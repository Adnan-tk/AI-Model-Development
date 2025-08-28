## Thunderbolt Chatbot API Reference

This document lists public classes, functions, and components with concise usage examples.

### Module: `app`

Class: `ChatbotApp`
- Methods: `show_header()`, `show_sidebar()`, `show_home()`, `show_file_upload()`, `process_data()`, `train_model()`, `chat_interface()`, `run()`
- Usage:
```python
from app import ChatbotApp
app = ChatbotApp()
app.run()
```

Run via UI:
```bash
streamlit run app.py
```

### Module: `data_processing.data_loader`

Class: `DataLoader(upload_folder="static/files")`
- `save_uploaded_file(uploaded_file) -> str`
- `load_csv(path, **kwargs) -> pd.DataFrame`
- `load_excel(path, sheet_name=0, **kwargs) -> pd.DataFrame`
- `load_json(path, **kwargs) -> dict`
- `load_txt(path, encoding='utf-8') -> str`
- `load_pickle(path) -> Any`
- `load_docx(path) -> str`
- `load_pdf(path) -> str`
- `load_huggingface_dataset(name, split='train', **kwargs) -> pd.DataFrame`
- `load_from_file(uploaded_file) -> Union[pd.DataFrame, str, dict, list]`
- `get_conversation_pairs(question_col=None, answer_col=None) -> list[tuple[str,str]]`

Example:
```python
from data_processing.data_loader import DataLoader
dl = DataLoader()
df = dl.load_csv("sample.csv")
pairs = dl.get_conversation_pairs("question", "answer")
```

### Module: `data_processing.preprocessor`

Class: `TextPreprocessor`
- `clean_text(text: str) -> str`
- `train_tokenizer(texts: list[str], vocab_size=30000, model_prefix='tokenizer') -> SentencePieceProcessor`
- `load_tokenizer(model_path: str) -> SentencePieceProcessor`
- `tokenize(texts: list[str], max_length=512) -> list[list[int]]`
- `decode(token_ids: list[int]) -> str`
- `split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)`

Example:
```python
from data_processing.preprocessor import TextPreprocessor
tp = TextPreprocessor()
sp = tp.train_tokenizer(["hello world", "goodbye"], vocab_size=8000)
ids = tp.tokenize(["hello world"])[0]
text = tp.decode(ids)
```

### Module: `model.transformer`

Classes:
- `PositionalEncoding(d_model, max_len=5000)`
- `MultiHeadAttention(d_model, num_heads, dropout=0.1)`
- `FeedForward(d_model, d_ff=2048, dropout=0.1)`
- `Transformer(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=512, dropout=0.1)`
- `DeepSeekChatbot(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=512, dropout=0.1)`

Key methods:
- `Transformer.forward(x, mask=None) -> torch.Tensor`
- `DeepSeekChatbot.forward(x, mask=None) -> torch.Tensor`
- `DeepSeekChatbot.generate(input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9) -> torch.Tensor`

Example (generation):
```python
import torch
from model.transformer import DeepSeekChatbot

model = DeepSeekChatbot(vocab_size=30000, d_model=256, num_heads=4, num_layers=4)
input_ids = torch.tensor([[2, 10, 42]])  # example BOS + tokens
generated = model.generate(input_ids, max_length=30)
```

### Module: `model.trainer`

Class: `ChatbotTrainer(model, vocab_size, learning_rate=1e-4, device=None)`
- `create_masks(src, tgt)`
- `train_epoch(dataloader) -> float`
- `validate(dataloader) -> float`
- `train(train_loader, val_loader, num_epochs=10, save_dir='checkpoints')`
- `load_model(checkpoint_path)`

Example:
```python
from model.trainer import ChatbotTrainer
trainer = ChatbotTrainer(model, vocab_size=30000)
trainer.train(train_loader, val_loader, num_epochs=5)
```

### Module: `utils.helpers`

Functions:
- `set_seed(seed=42)`
- `format_conversation(question, answer) -> str`
- `create_dataloader(inputs, targets, batch_size=32, shuffle=True) -> DataLoader`
- `preprocess_batch(questions, answers, tokenizer, max_length=512) -> (Tensor, Tensor)`

Alternate class (legacy): `ChatbotTrainer` — lower-level training loop similar to `model.trainer.ChatbotTrainer`.

### Module: `utils.visualization`

Class: `VisualizationUtils`
- `create_decision_tree() -> matplotlib.figure.Figure`
- `_draw_node(ax, x, y, text, color)`
- `_draw_connection(ax, x1, y1, x2, y2)`
- `plot_training_history(train_losses, val_losses) -> Figure`
- `plot_attention_weights(attention_weights) -> Figure`

### Module: `utils.ai_methods`

Class: `AIMethodologyExplainer`
- `show_methodology_info(method_key)`
- `show_comparison()`
- `show_methodology_selector()`

### Module: `train_chatbot`

Class: `ChatbotTrainer(model_name="microsoft/DialoGPT-medium")`
- Data loaders: `load_data_from_csv`, `load_data_from_json`, `load_data_from_txt`
- `tokenize_data(conversations) -> Dataset`
- `train(train_dataset, output_dir='./chatbot-model', epochs=3)`
- `generate_response(input_text, max_length=100) -> str`

Example:
```python
from train_chatbot import ChatbotTrainer
hf = ChatbotTrainer()
convs = hf.load_data_from_csv("conversations.csv")
tok = hf.tokenize_data(convs)
hf.train(tok, epochs=1)
print(hf.generate_response("Hello!"))
```

