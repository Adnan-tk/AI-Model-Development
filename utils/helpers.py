import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_conversation(question, answer):
    """Format conversation for training"""
    return f"<s>User: {question}\nAssistant: {answer}</s>"


def create_dataloader(inputs, targets, batch_size=32, shuffle=True):
    """Create a DataLoader from inputs and targets"""
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def preprocess_batch(questions, answers, tokenizer, max_length=512):
    """Preprocess a batch of questions and answers"""
    # Format conversations
    conversations = [format_conversation(q, a) for q, a in zip(questions, answers)]

    # Tokenize
    tokenized = tokenizer(conversations, max_length=max_length)

    # Create input and target sequences
    inputs = []
    targets = []

    for tokens in tokenized:
        inputs.append(tokens[:-1])  # All tokens except the last
        targets.append(tokens[1:])  # All tokens except the first

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime


class ChatbotTrainer:
    def __init__(self, model, vocab_size, learning_rate=0.0001, device=None):
        self.model = model
        self.vocab_size = vocab_size

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_losses = []
        self.val_losses = []

    def create_masks(self, src, tgt):
        """Create masks for transformer"""
        # Source padding mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # Target padding mask and look-ahead mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return src_mask, tgt_mask

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Training")):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Create masks
            src_mask, tgt_mask = self.create_masks(src, tgt)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, src_mask)

            # Calculate loss
            loss = self.criterion(output.view(-1, self.vocab_size), tgt.view(-1))

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Validating")):
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                # Create masks
                src_mask, tgt_mask = self.create_masks(src, tgt)

                # Forward pass
                output = self.model(src, src_mask)

                # Calculate loss
                loss = self.criterion(output.view(-1, self.vocab_size), tgt.view(-1))
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, num_epochs=10, save_dir='checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Step scheduler
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'vocab_size': self.vocab_size
                }, os.path.join(save_dir, 'best_model.pt'))

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'vocab_size': self.vocab_size
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

        print("Training completed!")

    def load_model(self, checkpoint_path):
        """Load a trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        print(f"Model loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")

