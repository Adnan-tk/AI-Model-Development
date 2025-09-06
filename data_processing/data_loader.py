import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import pickle
import docx
import PyPDF2
from io import BytesIO
import os
import re


class DataLoader:
    def __init__(self, upload_folder="static/files"):
        self.data = None
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to the server"""
        file_path = os.path.join(self.upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def load_csv(self, file_path, **kwargs):
        """Load data from CSV file"""
        self.data = pd.read_csv(file_path, **kwargs)
        return self.data

    def load_excel(self, file_path, sheet_name=0, **kwargs):
        """Load data from Excel file"""
        try:
            self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            return self.data
        except ImportError:
            raise ImportError(
                "Missing optional dependency 'openpyxl'. "
                "Use pip or conda to install openpyxl."
            )

    def load_json(self, file_path, **kwargs):
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def load_txt(self, file_path, encoding='utf-8'):
        """Load data from text file"""
        with open(file_path, 'r', encoding=encoding) as f:
            self.data = f.read()
        return self.data

    def load_pickle(self, file_path):
        """Load data from pickle file"""
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        return self.data

    def load_docx(self, file_path):
        """Load data from Word document"""
        try:
            doc = docx.Document(file_path)
            self.data = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self.data
        except ImportError:
            raise ImportError(
                "Missing optional dependency 'python-docx'. "
                "Use pip or conda to install python-docx."
            )

    def load_pdf(self, file_path):
        """Load data from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            self.data = text
            return self.data
        except ImportError:
            raise ImportError(
                "Missing optional dependency 'PyPDF2'. "
                "Use pip or conda to install PyPDF2."
            )

    def load_huggingface_dataset(self, dataset_name, split='train', **kwargs):
        """Load dataset from Hugging Face"""
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        self.data = pd.DataFrame(dataset)
        return self.data

    def load_from_file(self, uploaded_file):
        """Load data from any supported file type"""
        file_path = self.save_uploaded_file(uploaded_file)
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.csv':
                return self.load_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return self.load_excel(file_path)
            elif file_extension == '.json':
                return self.load_json(file_path)
            elif file_extension == '.txt':
                return self.load_txt(file_path)
            elif file_extension in ['.pkl', '.pickle']:
                return self.load_pickle(file_path)
            elif file_extension == '.docx':
                return self.load_docx(file_path)
            elif file_extension == '.pdf':
                return self.load_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            # Clean up the file if there was an error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError(f"Error loading file: {str(e)}")

    def get_conversation_pairs(self, question_col=None, answer_col=None):
        """Extract question-answer pairs from data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Handle different data types
        if isinstance(self.data, pd.DataFrame):
            if question_col is None or answer_col is None:
                # Try to auto-detect columns
                possible_question_cols = ['question', 'input', 'query', 'ask', 'user']
                possible_answer_cols = ['answer', 'response', 'output', 'reply', 'assistant']

                question_col = None
                answer_col = None

                for col in self.data.columns:
                    if any(word in col.lower() for word in possible_question_cols):
                        question_col = col
                    elif any(word in col.lower() for word in possible_answer_cols):
                        answer_col = col

                if question_col is None or answer_col is None:
                    # Use first two columns as fallback
                    question_col = self.data.columns[0]
                    answer_col = self.data.columns[1] if len(self.data.columns) > 1 else self.data.columns[0]

            return list(zip(self.data[question_col].astype(str).tolist(),
                            self.data[answer_col].astype(str).tolist()))
        elif isinstance(self.data, str):
            # For text data, split into sentences for question-answer pairs
            sentences = re.split(r'[.!?]', self.data)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            pairs = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    pairs.append((sentences[i], sentences[i + 1]))
            return pairs
        elif isinstance(self.data, list):
            # Assume list of (question, answer) tuples
            return self.data
        elif isinstance(self.data, dict):
            # Try to extract questions and answers from dictionary
            questions = []
            answers = []
            for key, value in self.data.items():
                if any(word in key.lower() for word in ['question', 'input', 'query', 'ask']):
                    if isinstance(value, list):
                        questions.extend(value)
                    else:
                        questions.append(value)
                elif any(word in key.lower() for word in ['answer', 'response', 'output', 'reply']):
                    if isinstance(value, list):
                        answers.extend(value)
                    else:
                        answers.append(value)

            if questions and answers:
                min_len = min(len(questions), len(answers))
                return list(zip(questions[:min_len], answers[:min_len]))
            else:
                # Try to find any key-value pairs that might be questions and answers
                pairs = []
                for key, value in self.data.items():
                    if isinstance(value, str) and len(value) > 10:
                        pairs.append((key, value))
                return pairs
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")
