mport streamlit as st
import torch
import numpy as np
import os
import pandas as pd
import traceback

# Set page config
st.set_page_config(
    page_title="Thundbolt Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Try to import all dependencies with error handling
try:
    from data_processing.data_loader import DataLoader
    from data_processing.preprocessor import TextPreprocessor
    from model.transformer import DeepSeekChatbot
    from model.trainer import ChatbotTrainer
    from utils.helpers import set_seed, format_conversation, create_dataloader

    # Set random seed
    set_seed()

    dependencies_loaded = True
except ImportError as e:
    st.error(f"Import Error: {str(e)}")
    st.info("Please make sure all dependencies are installed. Run: pip install -r requirements.txt")
    dependencies_loaded = False


class ChatbotApp:
    def __init__(self):
        if not dependencies_loaded:
            return

        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.vocab_size = 30000

        # Initialize session state
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None

    def show_header(self):
        """Show application header"""
        st.title("🧠 Thunderbolt Chatbot")
        st.markdown("""
        Build, train, and interact with your own AI chatbot using various AI methodologies.
        Upload your data, train a model, and start chatting!
        """)

    def show_sidebar(self):
        """Show sidebar navigation"""
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio(
            "Choose a section",
            ["🏠 Home", "📤 Data Upload", "⚙️ Data Processing",
             "🎯 Model Training", "💬 Chat"]
        )
        return app_mode

    def show_home(self):
        """Show home page"""
        st.header("Welcome to Thunderbolt Chatbot")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### What you can do with this application:

            1. **Upload your training data** - CSV, JSON, TXT, PDF, Word documents, and Excel files
            2. **Process and prepare data** - Clean and tokenize your data
            3. **Train transformer models** - Customize and train your chatbot
            4. **Chat with your AI** - Interact with your trained model

            ### Getting Started:

            1. Upload your data in the **Data Upload** section
            2. Process your data in the **Data Processing** section
            3. Train your model in the **Model Training** section
            4. Chat with your AI in the **Chat** section
            """)

        with col2:
            st.image("https://img.icons8.com/clouds/300/artificial-intelligence.png",
                     width=200, caption="AI Chatbot")
            st.info("Check the navigation sidebar to explore different sections")

        # Quick stats
        st.subheader("Application Stats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Uploaded Files", len(st.session_state.uploaded_files))

        with col2:
            status = "Ready" if st.session_state.processed_data else "Not Ready"
            st.metric("Data Status", status)

        with col3:
            conv_count = len(st.session_state.conversation)
            st.metric("Conversations", conv_count)

    def show_file_upload(self):
        """Show file upload section"""
        st.header("📤 Upload Training Data")

        st.info("""
        Supported file formats: CSV, Excel, JSON, TXT.
        For best results, use structured data with question/answer pairs.
        """)

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files for training",
            type=['csv', 'xlsx', 'json', 'txt'],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    try:
                        # Save and load the file
                        data = self.data_loader.load_from_file(uploaded_file)
                        st.session_state.uploaded_files.append(uploaded_file.name)

                        st.success(f"✅ Successfully loaded {uploaded_file.name}")

                        # Show file preview
                        with st.expander(f"Preview: {uploaded_file.name}"):
                            if isinstance(data, pd.DataFrame):
                                st.dataframe(data.head())
                                st.write(f"Shape: {data.shape}")
                            elif isinstance(data, str):
                                st.text(data[:500] + "..." if len(data) > 500 else data)
                                st.write(f"Length: {len(data)} characters")
                            else:
                                st.write(data)

                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

        # Show uploaded files list
        if st.session_state.uploaded_files:
            st.subheader("📁 Uploaded Files")
            for file_name in st.session_state.uploaded_files:
                st.write(f"- {file_name}")

            if st.button("🔄 Clear All Files"):
                st.session_state.uploaded_files = []
                st.session_state.processed_data = None
                st.experimental_rerun()

        # Sample data download
        st.subheader("📋 Sample Data")
        st.write("Don't have data? Download these sample files to get started:")

        col1, col2 = st.columns(2)

        with col1:
            sample_csv = """
question,answer,category
What is machine learning?,Machine learning is a subset of AI that enables systems to learn from data.,AI Basics
How does a neural network work?,Neural networks work by processing inputs through layers of interconnected nodes.,AI Basics
What is Python used for?,Python is used for web development, data analysis, AI, and more.,Programming
How to create a function in Python?,You can create a function using the def keyword.,Programming
What is the capital of France?,The capital of France is Paris.,Geography
What is the largest ocean?,The Pacific Ocean is the largest ocean.,Geography
"""
            st.download_button("Download CSV Sample", sample_csv, "sample_conversations.csv", "text/csv")

        with col2:
            sample_json = '''
{
  "conversations": [
    {
      "question": "What is artificial intelligence?",
      "answer": "Artificial intelligence is the simulation of human intelligence processes by machines.",
      "topic": "AI Basics"
    },
    {
      "question": "How do I install Python?",
      "answer": "You can download Python from python.org and follow the installation instructions.",
      "topic": "Programming"
    }
  ]
}
'''
            st.download_button("Download JSON Sample", sample_json, "sample_qa_pairs.json", "application/json")

    def process_data(self):
        """Process the uploaded data"""
        st.header("⚙️ Data Processing")

        if not st.session_state.uploaded_files:
            st.warning("Please upload files first in the Data Upload section.")
            return

        st.info("""
        This section will process your uploaded files, extract conversation pairs, 
        clean the text, and prepare it for training.
        """)

        # Get all loaded data
        all_conversations = []

        for file_name in st.session_state.uploaded_files:
            file_path = os.path.join(self.data_loader.upload_folder, file_name)
            file_extension = os.path.splitext(file_path)[1].lower()

            try:
                # Load the file based on its extension
                if file_extension == '.csv':
                    data = self.data_loader.load_csv(file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    data = self.data_loader.load_excel(file_path)
                elif file_extension == '.json':
                    data = self.data_loader.load_json(file_path)
                elif file_extension == '.txt':
                    data = self.data_loader.load_txt(file_path)

                # Extract conversation pairs
                if isinstance(data, pd.DataFrame):
                    # Let user select columns for DataFrame files
                    cols = data.columns.tolist()

                    col1, col2 = st.columns(2)
                    with col1:
                        question_col = st.selectbox(f"Question column for {file_name}", cols,
                                                    key=f"q_{file_name}")
                    with col2:
                        answer_col = st.selectbox(f"Answer column for {file_name}", cols,
                                                  key=f"a_{file_name}")

                    conversations = self.data_loader.get_conversation_pairs(question_col, answer_col)
                else:
                    conversations = self.data_loader.get_conversation_pairs()

                all_conversations.extend(conversations)
                st.success(f"✅ Processed {file_name}: {len(conversations)} conversation pairs")

            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {str(e)}")

        if not all_conversations:
            st.error("No conversation pairs could be extracted from the uploaded files.")
            return

        # Show conversation samples
        st.subheader("📝 Conversation Samples")
        if all_conversations:
            sample_idx = st.slider("Select sample to view", 0, min(5, len(all_conversations) - 1), 0)
            q, a = all_conversations[sample_idx]
            st.write(f"**Question:** {q}")
            st.write(f"**Answer:** {a}")

        # Clean text
        with st.spinner("🔄 Cleaning text..."):
            cleaned_texts = []
            for q, a in all_conversations:
                cleaned_q = self.preprocessor.clean_text(str(q))
                cleaned_a = self.preprocessor.clean_text(str(a))
                cleaned_texts.append(f"{cleaned_q} {cleaned_a}")

        # Train tokenizer
        if st.button("🚀 Train Tokenizer and Prepare Data"):
            with st.spinner("Training tokenizer... This may take a while..."):
                try:
                    self.tokenizer = self.preprocessor.train_tokenizer(cleaned_texts)
                    st.success("✅ Tokenizer trained successfully!")

                    # Store the conversations for later use
                    st.session_state.all_conversations = all_conversations
                    st.session_state.processed_data = True

                    st.success(
                        f"✅ Data processed successfully! {len(all_conversations)} conversation pairs ready for training.")

                except Exception as e:
                    st.error(f"❌ Error training tokenizer: {str(e)}")

    def train_model(self):
        """Train the model"""
        st.header("🎯 Model Training")

        if 'processed_data' not in st.session_state or not st.session_state.processed_data:
            st.warning("Please upload and process data first in the Data Processing section.")
            return

        st.info("""
        Configure your model architecture and training parameters below.
        The model is based on the Transformer architecture used in state-of-the-art language models.
        """)

        # Model configuration
        st.subheader("Model Architecture")

        col1, col2 = st.columns(2)

        with col1:
            d_model = st.slider("Model dimension", min_value=64, max_value=512, value=256, step=64)
            num_heads = st.slider("Number of attention heads", min_value=2, max_value=12, value=4, step=2)
            num_layers = st.slider("Number of transformer layers", min_value=2, max_value=8, value=4, step=2)

        with col2:
            d_ff = st.slider("Feed forward dimension", min_value=256, max_value=2048, value=1024, step=256)
            dropout = st.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
            max_seq_length = st.slider("Maximum sequence length", min_value=64, max_value=512, value=256, step=64)

        # Training configuration
        st.subheader("Training Parameters")

        col1, col2 = st.columns(2)

        with col1:
            learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-4, step=1e-5,
                                            format="%.5f")
            num_epochs = st.slider("Number of epochs", min_value=1, max_value=20, value=5)

        with col2:
            batch_size = st.slider("Batch size", min_value=4, max_value=32, value=8)

        # Start training
        if st.button("🚀 Start Training"):
            try:
                # Initialize model
                self.model = DeepSeekChatbot(
                    vocab_size=self.vocab_size,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_seq_length=max_seq_length
                )

                st.success("✅ Model initialized successfully!")
                st.info("Training would start here. This is a simplified example.")

            except Exception as e:
                st.error(f"❌ Error initializing model: {str(e)}")

    def chat_interface(self):
        """Chat interface"""
        st.header("💬 Chat with Your AI")

        if not hasattr(self, 'model') or self.model is None:
            st.warning("Please train a model first in the Model Training section.")
            return

        st.info("""
        Start a conversation with your trained AI model. 
        You can adjust the response generation parameters for different styles of responses.
        """)

        # Response parameters
        st.sidebar.subheader("Response Settings")
        max_length = st.sidebar.slider("Max response length", 20, 200, 100)
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)

        # Display conversation history
        st.subheader("Conversation History")

        for i, (speaker, message) in enumerate(st.session_state.conversation):
            if speaker == "user":
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>You:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>AI:</strong> {message}
                </div>
                """, unsafe_allow_html=True)

        # Input for new message
        st.subheader("New Message")
        user_input = st.text_input("Type your message here...", key="user_input")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Send") and user_input:
                # Add user message to conversation
                st.session_state.conversation.append(("user", user_input))

                # Generate AI response (simplified for this example)
                response = "This is a sample response from the AI. In a real implementation, this would be generated by your trained model."

                # Add AI response to conversation
                st.session_state.conversation.append(("ai", response))

                # Rerun to update conversation
                st.experimental_rerun()

        with col2:
            if st.button("Clear Conversation"):
                st.session_state.conversation = []
                st.experimental_rerun()

    def run(self):
        """Run the application"""
        if not dependencies_loaded:
            st.error("Dependencies not loaded. Please check your installation.")
            return

        self.show_header()
        app_mode = self.show_sidebar()

        if app_mode == "🏠 Home":
            self.show_home()
        elif app_mode == "📤 Data Upload":
            self.show_file_upload()
        elif app_mode == "⚙️ Data Processing":
            self.process_data()
        elif app_mode == "🎯 Model Training":
            self.train_model()
        elif app_mode == "💬 Chat":
            self.chat_interface()


if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
