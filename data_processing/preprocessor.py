import re
import pandas as pd
from sentencepiece import SentencePieceProcessor
from sklearn.model_selection import train_test_split
import numpy as np
import sentencepiece as spm


class TextPreprocessor:
    def __init__(self):
        self.sp_model = None
        self.vocab_size = 30000

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def train_tokenizer(self, texts, vocab_size=30000, model_prefix='tokenizer'):
        """Train a SentencePiece tokenizer"""
        # Save texts to temporary file
        with open('temp_texts.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        # Train tokenizer - REMOVE <unk> from user_defined_symbols
        spm.SentencePieceTrainer.train(
            input='temp_texts.txt',
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            user_defined_symbols=['<pad>', '<s>', '</s>'],  # REMOVED <unk> from here
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )

        # Load the trained model
        self.sp_model = SentencePieceProcessor()
        self.sp_model.load(f'{model_prefix}.model')
        self.vocab_size = vocab_size

        return self.sp_model

    def load_tokenizer(self, model_path):
        """Load a pre-trained tokenizer"""
        self.sp_model = SentencePieceProcessor()
        self.sp_model.load(model_path)
        return self.sp_model

    def tokenize(self, texts, max_length=512):
        """Tokenize texts using the trained tokenizer"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained. Call train_tokenizer first.")

        tokenized = []
        for text in texts:
            tokens = self.sp_model.encode_as_ids(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))  # Pad with zeros
            tokenized.append(tokens)

        return tokenized

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded.")
        return self.sp_model.decode_ids(token_ids)

    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=random_state
        )

        # Adjust validation size relative to temp size
        val_relative_size = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_relative_size, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test