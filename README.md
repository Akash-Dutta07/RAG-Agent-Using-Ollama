# RAG Project with Ollama & ChromaDB

A Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content using local AI models.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF files for question-answering
- **Local AI Models**: Uses Ollama for both embeddings and language generation
- **Vector Search**: ChromaDB for efficient document retrieval
- **Interactive Web Interface**: Streamlit-based user interface
- **Privacy-First**: All processing happens locally - no data sent to external APIs

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Language Model**: Llama 3.2:1b (via Ollama)
- **Embeddings**: nomic-embed-text (via Ollama)
- **Vector Database**: ChromaDB
- **Document Processing**: LangChain
- **Package Management**: uv

## 📋 Prerequisites

Before running this application, make sure you have:

1. **Python 3.12+** installed
2. **Ollama** installed and running
3. **uv** package manager installed

### Installing Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### Installing uv

```bash
# On Windows (PowerShell)
curl -LsSf https://astral.sh/uv/install.ps1 | powershell

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Akash-Dutta07/RAG-Agent-Using-Ollama.git
cd RAG-Agent-Using-Ollama/project
```

### 2. Set Up Python Environment

```bash
# Initialize uv project (if not already done)
uv init

# Install dependencies
uv add -r requirements.txt

# Create lock file and sync
uv lock
uv sync
```

### 3. Download Required Ollama Models

```bash
# Download the language model
ollama pull llama3.2:1b

# Download the embedding model
ollama pull nomic-embed-text
```

### 4. Verify Models are Downloaded

```bash
ollama list
```

You should see both `llama3.2:1b` and `nomic-embed-text` in the list.

### 5. Run the Application

```bash
# Make sure you're in the project directory
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 💻 Usage

1. **Upload PDF**: Click "Browse files" or drag and drop a PDF file
2. **Process Document**: Click the "Process" button to create embeddings
3. **Ask Questions**: Enter your questions in the text area and click "Get Answer"
4. **View Context**: Expand "Show Context" to see the source text used for the answer

## 📁 Project Structure

```
project/
├── main.py                 # Streamlit application
├── supporting_functions.py # RAG implementation
├── requirements.txt        # Python dependencies
├── pyproject.toml         # uv project configuration
└── README.md              # This file
```

## 🔧 Configuration

You can modify the models used by editing `supporting_functions.py`:

```python
EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model
LLM_MODEL = "llama3.2:1b"            # Language model
```

### Available Ollama Models

- **Language Models**: `llama3.2:1b`, `llama3.2:3b`, `qwen2.5:7b`, etc.
- **Embedding Models**: `nomic-embed-text`, `mxbai-embed-large`, etc.

## 📚 Dependencies

Key dependencies include:

- `streamlit`: Web interface
- `langchain`: Document processing framework
- `langchain-community`: Community extensions
- `langchain-ollama`: Ollama integration
- `langchain-text-splitters`: Text chunking
- `chromadb`: Vector database
- `pypdf`: PDF processing
- `ollama`: Ollama client

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   ```bash
   # Download the missing model
   ollama pull <model-name>
   ```

2. **Ollama connection error**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama if not running
   ollama serve
   ```

3. **Import errors**
   ```bash
   # Reinstall dependencies
   uv sync --force
   ```

4. **PDF processing errors**
   - Ensure PDF is not corrupted
   - Check file size (limit: 200MB)
   - Try with a different PDF

⭐ If you found this project helpful, please give it a star on GitHub!
