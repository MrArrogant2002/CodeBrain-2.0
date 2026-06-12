# CodeBrain - BMS Code Analysis System
## Complete Project Documentation

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Component Details](#component-details)
6. [How It Works](#how-it-works)
7. [Setup & Installation](#setup--installation)
8. [Usage Guide](#usage-guide)
9. [Dependencies Explained](#dependencies-explained)

---

## Project Overview

**CodeBrain** is an intelligent code analysis and question-answering system specifically designed for analyzing Battery Management System (BMS) codebases. It combines semantic code understanding with large language models to answer domain-specific questions about C code repositories.

### Key Capabilities
- **Semantic Code Search**: Finds relevant code sections based on query meaning, not just keywords
- **Intelligent Context Retrieval**: Uses vector embeddings to extract the most relevant code snippets
- **LLM-Powered Analysis**: Leverages CodeLlama to reason about code and provide detailed answers
- **User-Friendly Interface**: PyQt5-based GUI for easy interaction with the system

### Target Use Cases
- Analyzing BMS (Battery Management System) codebases
- Understanding battery charging and monitoring logic
- Diagnosing voltage regulation and thermal control implementations
- Code comprehension and documentation generation

---

## Architecture

The CodeBrain system follows a **Retrieval-Augmented Generation (RAG)** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (PyQt5)                  │
│                    gui/app.py                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         ┌──────▼──────┐       ┌─────▼──────┐
         │ Query Input │       │   Answer   │
         │   Display   │       │   Display  │
         └──────┬──────┘       └────▲───────┘
                │                   │
                │                   │
         ┌──────▼──────────────────┴──────┐
         │   Query Processing Engine      │
         │       (main.py)                │
         └──────┬───────────────────┬─────┘
                │                   │
        ┌───────▼────────────┐  ┌──▼──────────────┐
        │ Embedding Engine   │  │  Vector Search  │
        │ (embedder.py)      │  │  (vector_store) │
        └───────┬────────────┘  └──▲──────────────┘
                │                  │
                │ embeddings       │ k-nearest
                │                  │ neighbors
        ┌───────▼──────────────────┴──────┐
        │   Vector Store (FAISS Index)    │
        │   with Embedded Code Chunks     │
        └───────┬──────────────────┬──────┘
                │                  │
        ┌───────▼────────────┐ ┌──▼──────────────┐
        │  LLM Reasoner      │ │ Retrieved Code  │
        │  (llm_reasoner.py) │ │ Context         │
        └───────┬────────────┘ └─────────────────┘
                │
                │ prompt + context
        ┌───────▼──────────────┐
        │  Ollama (CodeLlama)  │
        │  Local LLM Server    │
        └──────────────────────┘
```

### Data Flow
1. **Loading**: Repository loader scans and extracts C code files
2. **Embedding**: Code chunks are converted to vector embeddings using sentence-transformers
3. **Indexing**: Embeddings are indexed using FAISS for fast similarity search
4. **Query**: User question is embedded and used to search for similar code
5. **Retrieval**: Top-k most relevant code chunks are retrieved
6. **Generation**: Retrieved context + query is sent to CodeLlama for reasoning
7. **Response**: LLM generates an informed answer about the code

---

## Technology Stack

| Component | Technology | Purpose | Why Used |
|-----------|-----------|---------|----------|
| **GUI Framework** | PyQt5 | User interface | Cross-platform desktop app, robust widget system |
| **Vector Embeddings** | sentence-transformers | Convert code to vectors | Pre-trained on semantic similarity, fast inference |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Fast similarity search | Efficient k-NN search in high dimensions, CPU-friendly |
| **LLM Engine** | Ollama + CodeLlama | Code understanding & reasoning | Local LLM, specialized for code, privacy-first |
| **Deep Learning** | PyTorch | Underlying tensor operations | Industry standard, efficient computation |
| **NLP Pipeline** | Transformers (Hugging Face) | Token processing | State-of-the-art NLP models |
| **ML Utilities** | scikit-learn, scipy | Data processing & math | Data transformation, statistical operations |
| **Code Parsing** | Python standard library | File I/O & traversal | Simple, built-in C file extraction |

### Why This Stack?

- **Offline-First**: Uses local Ollama + CodeLlama instead of cloud APIs (privacy, cost, latency)
- **Lightweight**: FAISS is CPU-based, no GPU required for vector search
- **Semantic Understanding**: sentence-transformers captures meaning beyond keywords
- **Extensible**: Easy to add new C files or swap embedding models
- **Cost-Effective**: No API calls, no per-query fees

---

## Project Structure

```
CODEBRAIN/
├── main.py                          # Main entry point (CLI demo)
├── requirements.txt                 # Python dependencies
├── doc.md                           # This file
│
├── gui/                             # GUI module
│   ├── app.py                       # PyQt5 GUI application
│   ├── main.py                      # Re-exports query_system
│   │
│   ├── engine/                      # Core reasoning engine
│   │   ├── repo_loader.py          # Loads C files from repository
│   │   ├── embedder.py             # Generates vector embeddings
│   │   ├── vector_store.py         # FAISS-based vector index
│   │   ├── llm_reasoner.py         # Ollama integration
│   │   │
│   │   └── sample_bms_repo/        # Sample Battery Management System code
│   │       ├── battery_monitor.c   # Battery voltage monitoring
│   │       ├── charging_controller.c # Charging logic
│   │       └── temperature_control.c # Thermal management
│
└── .gitignore / README.md           # (Standard project files)
```

### File Responsibilities

| File | Responsibility | Key Functions |
|------|-----------------|----------------|
| `main.py` | CLI entry point, system orchestration | `query_system()` |
| `gui/app.py` | PyQt5 GUI with text input/output | `CodeBrainGUI` class |
| `gui/engine/repo_loader.py` | Scans and loads C files | `load_c_files()` |
| `gui/engine/embedder.py` | Creates vector embeddings | `generate_embeddings()` |
| `gui/engine/vector_store.py` | Manages vector index | `VectorStore` class with add/search |
| `gui/engine/llm_reasoner.py` | Queries Ollama LLM | `ask_llm()` |

---

## Component Details

### 1. Repository Loader (`repo_loader.py`)

**Purpose**: Extract all C source files from a repository into analyzable chunks.

```python
def load_c_files(repo_path) -> List[Dict]
```

**How it works:**
- Recursively walks through directory tree
- Identifies files ending with `.c`
- Reads each file and stores: filename + code content
- Handles encoding errors gracefully

**Why this approach:**
- Simple, no external parsing library needed
- Treats whole files as chunks (preserves context)
- Extensible to other file types (`.h`, `.cpp`, etc.)

**Data Structure:**
```python
[
  {"file": "battery_monitor.c", "code": "// C source code..."},
  {"file": "charging_controller.c", "code": "// C source code..."},
  # ... more files
]
```

---

### 2. Embedder (`embedder.py`)

**Purpose**: Convert text code into numerical vector representations that capture semantic meaning.

```python
def generate_embeddings(chunks) -> np.ndarray
```

**Model Used**: `all-MiniLM-L6-v2` (Sentence Transformers)
- Generates 384-dimensional vectors
- Trained on semantic similarity tasks
- Lightweight (22MB) but effective for code

**Why sentence-transformers?**
- Pre-trained on similarity tasks (good for semantic code search)
- Fast inference (~1ms per chunk on CPU)
- Works well with specialized code (not just natural language)
- Can be swapped for domain-specific models (e.g., `code-embeddings-384`)

**Output:**
- NumPy array of shape (n_chunks, 384) where each row is a code chunk's embedding

---

### 3. Vector Store (`vector_store.py`)

**Purpose**: Store embeddings in an efficient searchable index and retrieve similar code chunks.

```python
class VectorStore:
    def add(embeddings, chunks) -> None        # Index embeddings
    def search(query_embedding, k=3) -> List   # Retrieve top-k similar
```

**Technology**: FAISS (Facebook AI Similarity Search)
- Uses flat L2 (Euclidean) distance metric
- Finds k-nearest neighbors in vector space

**Why FAISS?**
- **Fast**: O(n) search but with optimized SIMD operations
- **Scalable**: Can handle millions of vectors
- **Simple**: IndexFlatL2 requires no training
- **CPU-friendly**: No GPU needed; works on laptops

**How search works:**
1. Query question is embedded to 384-dim vector
2. FAISS computes L2 distance to all indexed vectors
3. Returns indices of k closest vectors
4. Original code chunks retrieved by index

**Trade-offs:**
- Doesn't require GPU but could use one if available
- Flat index fine for small-medium repos (~1000s of files)
- Could upgrade to `IndexIVFFlat` for very large repos

---

### 4. LLM Reasoner (`llm_reasoner.py`)

**Purpose**: Query a local LLM to understand code and answer questions about it.

```python
def ask_llm(prompt) -> str
```

**LLM Used**: CodeLlama (via Ollama)
- 7B, 13B, or 34B parameter variants available
- Specialized for code understanding (trained on ~1 trillion tokens of code)
- Runs 100% locally (no data leaves your machine)

**Why Ollama + CodeLlama?**

| Aspect | Alternative | Why CodeLlama + Ollama |
|--------|-------------|----------------------|
| **Privacy** | OpenAI API, Claude | All code stays local |
| **Cost** | API pricing | Free (one-time download) |
| **Latency** | Cloud roundtrip | Direct local inference |
| **Dependency** | API uptime | Works offline |
| **Control** | Black-box inference | Full model access |

**Prompt Structure:**
```
Question: <user_query>
Code:
File: battery_monitor.c
<retrieved_code_chunk_1>

File: charging_controller.c
<retrieved_code_chunk_2>

// ... more chunks
```

**Error Handling:**
- Detects Ollama connection failures
- Provides helpful error message with setup instructions
- Graceful fallback instead of crashes

---

### 5. GUI Application (`gui/app.py`)

**Purpose**: Provide a user-friendly interface for querying the code analysis system.

**Framework**: PyQt5
- Cross-platform (Windows, Linux, macOS)
- Native-looking UIs
- Rich widget library

**UI Components:**
- **Input TextEdit**: User enters diagnostic questions
- **Analyze Button**: Triggers query execution
- **Output TextEdit**: Displays LLM response

**Why PyQt5?**
- Mature and stable
- Good for rapid GUI development
- Better performance than Tkinter
- Rich widget ecosystem

**User Flow:**
1. User types question (e.g., "How does battery charging work?")
2. Clicks "Analyze" button
3. System retrieves relevant code
4. LLM generates response
5. Response displayed in output area

---

## How It Works

### Complete Query Flow (Step-by-Step)

**Example**: User asks "How does the battery charging work?"

#### Step 1: System Initialization (main.py)
```python
# Load all .c files from repository
chunks = load_c_files("gui/sample_bms_repo")
# Output: [
#   {"file": "battery_monitor.c", "code": "..."},
#   {"file": "charging_controller.c", "code": "..."},
# ]

# Generate embeddings for each chunk
embeddings = generate_embeddings(chunks)
# Output: shape (3, 384) - 3 files, 384-dim vectors

# Initialize vector database
vector_db = VectorStore(dimension=384)
vector_db.add(embeddings, chunks)
```

#### Step 2: User Query (gui/app.py)
```python
# User types and clicks "Analyze"
question = "How does battery charging work?"
```

#### Step 3: Embedding Query
```python
# Embed the question to 384-dim vector
q_embed = embed_model.encode([question])
# Output: shape (1, 384)

# The embedding captures semantic meaning:
# "How does battery charging work?" 
# → vector that's close to charging_controller.c in vector space
```

#### Step 4: Retrieve Relevant Code
```python
# Search for top-3 most similar code chunks
results = vector_db.search(q_embed, k=3)

# FAISS internally:
# 1. Computes L2 distance: dist = ||q_embed - all_embeddings||
# 2. Finds indices with smallest distances
# 3. Returns corresponding code chunks

# Output:
# [
#   {"file": "charging_controller.c", "code": "void start_charging()..."},
#   {"file": "battery_monitor.c", "code": "void check_overvoltage()..."},
#   # ... (3rd most relevant chunk)
# ]
```

#### Step 5: Build Prompt
```python
# Construct prompt with question + context
prompt = f"""Question: How does battery charging work?
Code:
File: charging_controller.c
void start_charging() { ... }

File: battery_monitor.c
void check_overvoltage() { ... }

... more code ...
"""
```

#### Step 6: Query LLM
```python
# Send prompt to CodeLlama via Ollama
response = ask_llm(prompt)

# CodeLlama analyzes the code and generates:
# "Based on the provided code, battery charging works as follows:
#  1. The charging controller calls start_charging()...
#  2. The battery monitor checks for overvoltage...
#  ..."
```

#### Step 7: Display Result
```python
# GUI shows the response
self.output.setText(response)
```

### Why This Architecture Works

| Challenge | Solution | Benefit |
|-----------|----------|---------|
| Keyword search misses semantic meaning | Vector embeddings capture code intent | "charge battery" matches even if code says "start_charging()" |
| Sending whole codebase to LLM is slow | Retrieve only relevant chunks | Fast response, cheaper inference |
| LLM needs context to understand code | Include code snippets in prompt | Contextual, grounded answers |
| Manual code comprehension is tedious | Automation via ML + LLM | Scales to large codebases |

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- CodeLlama model pulled in Ollama

### Step 1: Install Python Dependencies
```bash
cd CODEBRAIN
pip install -r requirements.txt
```

### Step 2: Install & Configure Ollama

**Download Ollama:**
- macOS: `brew install ollama`
- Linux: `curl https://ollama.ai/install.sh | sh`
- Windows: Download from https://ollama.ai

**Pull CodeLlama model:**
```bash
ollama pull codellama
```

**Start Ollama server:**
```bash
ollama serve
# Server runs on http://localhost:11434 by default
```

### Step 3: Prepare Your Codebase

Replace sample BMS code with your actual C files:
```
gui/sample_bms_repo/
├── your_file_1.c
├── your_file_2.c
└── your_file_3.c
```

### Step 4: Run the Application

**Option A: GUI Mode**
```bash
cd gui
python app.py
```

**Option B: CLI Mode**
```bash
python main.py
```

---

## Usage Guide

### CLI Mode (main.py)

```bash
python main.py
```

**What happens:**
1. Loads all `.c` files from `gui/sample_bms_repo/`
2. Generates embeddings
3. Asks a hardcoded question: "How does the battery charging work?"
4. Prints response to console

**Good for:**
- Testing the system
- Batch processing queries
- Integration with other scripts

### GUI Mode (gui/app.py)

```bash
cd gui
python app.py
```

**Interface:**
```
┌─────────────────────────────────────────────┐
│     CodeBrain BMS Analyzer                  │
├─────────────────────────────────────────────┤
│ Enter BMS diagnostic question               │
│ ┌─────────────────────────────────────────┐ │
│ │ [Text input area]                       │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│              [ Analyze ]                     │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │ [Response displayed here]               │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Usage:**
1. Type your question in the input area
2. Click "Analyze"
3. Wait for response (typically 5-30 seconds depending on CodeLlama variant)
4. Read the analysis in the output area

**Example Queries:**
- "How does battery charging work?"
- "What happens during overvoltage detection?"
- "Explain the temperature control mechanism"
- "What files handle battery monitoring?"

---

## Dependencies Explained

### Core Dependencies

#### `PyQt5==5.15.9`
- **What**: GUI framework
- **Why**: Cross-platform desktop application
- **Size**: ~100 MB
- **Used in**: `gui/app.py` for user interface

#### `sentence-transformers>=3.0.0`
- **What**: Pre-trained embedding models
- **Why**: Convert code text to semantic vectors
- **Model used**: `all-MiniLM-L6-v2` (22 MB, 384-dim)
- **Used in**: `gui/engine/embedder.py`

#### `faiss-cpu>=1.8.0`
- **What**: Vector similarity search library
- **Why**: Fast k-NN search in high-dimensional space
- **Optimization**: CPU-optimized (no GPU needed)
- **Used in**: `gui/engine/vector_store.py`

#### `ollama`
- **What**: Python client for Ollama LLM server
- **Why**: Interface to local CodeLlama model
- **Note**: Ollama server must be running separately
- **Used in**: `gui/engine/llm_reasoner.py`

### ML/DL Support Libraries

#### `torch>=2.4.0`
- **What**: Deep learning framework
- **Why**: Underlying tensor operations for embeddings
- **Why this version**: Latest stable with good performance
- **Size**: ~300-600 MB depending on variant
- **Note**: Included with sentence-transformers

#### `transformers>=4.40.0`
- **What**: Hugging Face transformer models
- **Why**: NLP pipeline and tokenization
- **Used by**: sentence-transformers internally
- **Size**: ~1 GB for full library

#### `scikit-learn>=1.4.2`
- **What**: Machine learning utilities
- **Why**: Data processing, normalization
- **Size**: ~50 MB
- **Potential use**: Cross-validation, metrics

#### `scipy>=1.13.0`
- **What**: Scientific computing library
- **Why**: Matrix operations, statistics
- **Size**: ~50 MB
- **Potential use**: Distance calculations, optimization

#### `peft>=0.4.0`
- **What**: Parameter-Efficient Fine-Tuning
- **Why**: For future model fine-tuning capabilities
- **Note**: Currently not used but prepared for extensibility
- **Size**: ~1 MB

#### `numpy<2`
- **What**: Numerical computing
- **Why**: Underlying array operations for all ML libraries
- **Constraint**: `<2` because transformers/torch compatibility
- **Size**: ~10 MB
- **Used in**: `vector_store.py` for array operations

#### `tqdm>=4.66.2`
- **What**: Progress bar library
- **Why**: User-friendly progress visualization
- **Used in**: Potentially in batch processing (not currently)
- **Size**: <1 MB

### Dependency Installation

**Full installation:**
```bash
pip install -r requirements.txt
```

**Minimal installation (if space is critical):**
```bash
pip install PyQt5 sentence-transformers faiss-cpu ollama torch transformers
```

### Dependency Graph

```
CodeBrain
├── PyQt5 (GUI)
├── sentence-transformers (embeddings)
│   ├── torch (deep learning)
│   ├── transformers (NLP)
│   ├── scikit-learn
│   └── numpy
├── faiss-cpu (vector search)
│   └── numpy
├── ollama (LLM client)
└── scipy (math utilities)
```

### Version Rationale

| Package | Version | Why |
|---------|---------|-----|
| PyQt5 | Fixed `5.15.9` | Stability, known compatibility |
| torch | `>=2.4.0` | Latest features, good performance |
| transformers | `>=4.40.0` | Newer models, security updates |
| numpy | `<2` | API stability with older libs |
| others | `>=X.Y.Z` | Ensure minimum features |

---

## Performance Considerations

### System Requirements

**Minimum:**
- CPU: 2+ cores
- RAM: 4 GB
- Disk: 2 GB (embeddings) + 4-8 GB (CodeLlama model)

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB (for large codebases)
- Disk: 10+ GB (multiple model variants)

### Inference Times

| Operation | Time | Hardware |
|-----------|------|----------|
| Load & embed 10 files | ~1 second | CPU |
| Vector search (k=3) | ~1-2 ms | CPU |
| CodeLlama generation | 5-30 sec | CPU (7B model) |

**Bottleneck**: LLM inference is slowest; embedding & search are near-instant.

---

## Future Enhancements

### Potential Features

1. **Persistent Storage**
   - Save embeddings to disk to skip re-indexing
   - Speed up cold starts

2. **Model Selection**
   - Support multiple embedding models
   - Swap CodeLlama for other LLMs (GPT-4, Llama 2, etc.)

3. **Code Parsing**
   - Parse C syntax tree instead of treating files as text
   - Better semantic understanding of functions, structs

4. **Multi-Language Support**
   - Extend beyond C to C++, Python, Rust, etc.

5. **Interactive Debugging**
   - Highlight relevant code in response
   - Show retrieval confidence scores

6. **Fine-Tuning**
   - Fine-tune embeddings on domain-specific code
   - Improve retrieval relevance

7. **Web Interface**
   - FastAPI backend
   - React frontend for browser-based access

---

## Troubleshooting

### Issue: "Error querying ollama"

**Cause**: Ollama server not running

**Solution**:
```bash
ollama serve
# In another terminal
python gui/app.py
```

### Issue: "Model 'codellama' not found"

**Cause**: CodeLlama not downloaded

**Solution**:
```bash
ollama pull codellama
# Wait for download to complete (~4 GB)
```

### Issue: Slow embedding generation

**Cause**: Using large embedding model

**Solution**: Use lighter model
```python
# In embedder.py, change:
model = SentenceTransformer("all-MiniLM-L6-v2")  # Current (fast)
# to
model = SentenceTransformer("all-mpnet-base-v2")  # More accurate but slower
```

### Issue: High memory usage

**Cause**: Large codebase loaded entirely

**Solution**: Implement chunking in `repo_loader.py`:
```python
# Split large files into smaller chunks
# instead of treating whole file as one chunk
```

---

## Summary

**CodeBrain** is a smart code analysis system that:

1. **Loads** C code from a repository
2. **Embeds** code chunks into semantic vectors
3. **Indexes** vectors for fast similarity search
4. **Retrieves** relevant code based on queries
5. **Reasons** about code using a local LLM
6. **Responds** to user questions with informed answers

**Key Innovation**: Combines semantic search (embeddings) + reasoning (LLM) to understand code at a higher level than keyword search alone.

**Technology**: Modern ML stack (transformers, FAISS, Ollama) designed for privacy, speed, and cost-effectiveness.

**Extensibility**: Easy to adapt for other programming languages, codebases, or domains.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Install dependencies |
| `ollama serve` | Start LLM server |
| `ollama pull codellama` | Download CodeLlama |
| `python main.py` | Run CLI demo |
| `python gui/app.py` | Run GUI application |

---

**Documentation Version**: 1.0  
**Last Updated**: 2026-06-12  
**Project**: CodeBrain - BMS Code Analysis System
