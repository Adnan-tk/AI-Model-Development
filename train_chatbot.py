import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os


class ChatbotTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load_data_from_csv(self, file_path, question_col="question", answer_col="answer"):
        """Load training data from CSV file"""
        df = pd.read_csv(file_path)
        conversations = []

        for _, row in df.iterrows():
            conversation = f"<s>User: {row[question_col]}\nAssistant: {row[answer_col]}</s>"
            conversations.append(conversation)

        return conversations

    def load_data_from_json(self, file_path, question_key="question", answer_key="answer"):
        """Load training data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        conversations = []
        for item in data.get('conversations', []):
            conversation = f"<s>User: {item[question_key]}\nAssistant: {item[answer_key]}</s>"
            conversations.append(conversation)

        return conversations

    def load_data_from_txt(self, file_path):
        """Load training data from text file with User: and AI: prefixes"""
        conversations = []
        current_question = None

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('User:'):
                    current_question = line.replace('User:', '').strip()
                elif line.startswith('AI:') and current_question:
                    answer = line.replace('AI:', '').strip()
                    conversation = f"<s>User: {current_question}\nAssistant: {answer}</s>"
                    conversations.append(conversation)
                    current_question = None

        return conversations

    def tokenize_data(self, conversations):
        """Tokenize the conversation data"""

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )

        # Create dataset
        dataset = Dataset.from_dict({'text': conversations})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        return tokenized_dataset

    def train(self, train_dataset, output_dir="./chatbot-model", epochs=3):
        """Train the model"""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_dir='./logs',
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Training completed! Model saved to {output_dir}")

    def generate_response(self, input_text, max_length=100):
        """Generate a response to input text"""
        # Format the input
        input_text = f"<s>User: {input_text}\nAssistant:"

        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True
            )

        # Decode and return the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(input_text, "").strip()

        return response


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ChatbotTrainer()

    # Load data from different sources
    csv_conversations = trainer.load_data_from_csv("conversations.csv")
    json_conversations = trainer.load_data_from_json("qa_pairs.json")
    txt_conversations = trainer.load_data_from_txt("dialogues.txt")

    # Combine all conversations
    all_conversations = csv_conversations + json_conversations + txt_conversations
    print(f"Total training examples: {len(all_conversations)}")

    # Tokenize data
    tokenized_data = trainer.tokenize_data(all_conversations)

    # Train the model
    trainer.train(tokenized_data, epochs=3)

    # Test the model
    test_question = "What is machine learning?"
    response = trainer.generate_response(test_question)
    print(f"Q: {test_question}")
    print(f"A: {response}")