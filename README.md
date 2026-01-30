# RAG Application with Evaluation Framework

A Retrieval-Augmented Generation (RAG) web application built with LlamaIndex and FastAPI, including a baseline evaluation framework for measuring system quality.

## Overview

This application enhances a large language model with RAG capabilities, allowing it to reference an external knowledge base for more accurate and contextually relevant responses. It includes:

- **RAG Web Application**: Upload documents, query them via a chat interface
- **Evaluation Framework**: Measure retrieval and answer quality using keyword-based metrics

## End-to-End Setup

### Step 1: Clone and Install

```bash
git clone https://github.com/pandysp/RAG-eval.git
cd RAG-eval

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Requirements.txt
```

### Step 2: Configure API Key

```bash
cp openai_key.env.example openai_key.env
```

Edit `openai_key.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 3: Download Evaluation PDFs

Download the evaluation dataset from Google Drive:

**[Download Dataset](https://drive.google.com/drive/folders/18q_zokgsrMsL-Xfx4OcYST1DLb8TNzYY)**

The folder contains:
- `pdfs-20250915T225624Z-1-001.zip` - PDF files (part 1)
- `pdfs-20250915T225624Z-1-002.zip` - PDF files (part 2)
- `All_PDFs.csv` - PDF metadata
- `Eval_data.csv` - Full evaluation dataset

Download both zip files and extract them:
```bash
mkdir pdfs
unzip pdfs-20250915T225624Z-1-001.zip -d pdfs/
unzip pdfs-20250915T225624Z-1-002.zip -d pdfs/
```

### Step 4: Start the Server

```bash
uvicorn main:app --reload
```

The server will start at http://127.0.0.1:8000

### Step 5: Ingest PDFs

You have two options to ingest the PDFs:

**Option A: Web Interface**
1. Open http://127.0.0.1:8000 in your browser
2. Click "Choose File" and select all PDFs
3. Click "Upload Files"
4. Wait for ingestion to complete

**Option B: API (for bulk upload)**
```bash
# Using curl to upload all PDFs
for pdf in pdfs/*.pdf; do
  curl -X POST "http://127.0.0.1:8000/ingest" -F "files=@$pdf"
done
```

> **Note**: The first time you ingest documents, the system will create a vector index in `storage/`. This may take several minutes for 200 PDFs.

### Step 6: Run Evaluation

```bash
# Extract keywords from expected answers (one-time, ~$0.10)
python extract_keywords.py

# Run the evaluation
python evaluate.py
```

## Evaluation Results

The evaluation measures three metrics:

| Metric | Baseline Value | Description |
|--------|----------------|-------------|
| **Retrieval Hit Rate** | 98.0% | Correct source PDF in retrieved chunks |
| **Avg Keyword Match** | 58.7% | Expected keywords found in answer |
| **Answer Correctness** | 62.0% | Answers with ≥50% keyword match |

Detailed per-question results are saved to `eval_results.csv`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface for document upload and querying |
| `/ingest` | POST | Upload and index documents (PDF, TXT) |
| `/query` | GET | Query the knowledge base |
| `/query_with_context` | GET | Query with source documents (for evaluation) |

### Example API Usage

```bash
# Query the system
curl "http://127.0.0.1:8000/query?query=What%20is%20Domar%20aggregation"

# Query with source context (for evaluation)
curl "http://127.0.0.1:8000/query_with_context?query=What%20is%20Domar%20aggregation"
```

## Project Structure

```
RAG-eval/
├── main.py                  # FastAPI application with RAG endpoints
├── chat_interface.html      # Web UI for upload and chat
├── evaluate.py              # Evaluation script (keyword matching)
├── extract_keywords.py      # One-time keyword extraction from answers
├── Eval_data_subset.csv     # 200 evaluation questions with expected answers
├── eval_results.csv         # Output: per-question evaluation results
├── Requirements.txt         # Python dependencies
├── openai_key.env.example   # Template for API key configuration
├── storage/                 # Vector index (created on first ingest)
├── data/                    # Sample text documents
├── sample/                  # Demo files for testing
└── Images/                  # Screenshots for documentation
```

## How It Works

### RAG Pipeline

```
[PDF Upload] → [Chunking] → [Embedding] → [Vector Store]
                  ↓
              512 tokens
              50 overlap
                  ↓
         BAAI/bge-small-en-v1.5
                  ↓
            storage/*.json
```

```
[User Query] → [Embed Query] → [Retrieve Top-K] → [LLM Summarize] → [Response]
```

### Evaluation Pipeline

```
[Eval_data_subset.csv]
        ↓
[extract_keywords.py]  ←── One-time: LLM extracts 2-5 keywords per answer
        ↓
[Eval_data_subset.csv + keywords column]
        ↓
[evaluate.py]  ←── For each question:
        ↓           1. Query RAG system
        ↓           2. Check if source PDF in retrieved chunks (hit rate)
        ↓           3. Check keyword overlap (no LLM needed)
        ↓
[eval_results.csv]  ←── Per-question: query, answer, scores
        ↓
[Console Output]    ←── Aggregate metrics
```

## Configuration

Key settings in `main.py`:

```python
# Embedding model (local, no API calls)
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

# Chunking parameters
Settings.chunk_size = 512      # tokens per chunk
Settings.chunk_overlap = 50    # overlap between chunks
```

## Data Files

| File | Description |
|------|-------------|
| `Eval_data_subset.csv` | 200 questions with expected answers, source PDFs, and extracted keywords |
| `eval_results.csv` | Evaluation output with per-question scores |
| `All_PDFs.csv` | Full list of available PDFs with metadata |
| `Eval_data.csv` | Complete evaluation dataset (before subsetting) |

## Troubleshooting

**"No module named 'llama_index'"**
```bash
pip install -r Requirements.txt
```

**"OPENAI_API_KEY not set"**
```bash
cp openai_key.env.example openai_key.env
# Edit the file and add your API key
```

**Slow ingestion**
- First-time ingestion builds the vector index
- Subsequent runs load from `storage/` (fast)
- Delete `storage/` folder to rebuild index from scratch

**Low evaluation scores**
- Ensure all PDFs are ingested before running evaluation
- Check `eval_results.csv` for per-question analysis
- Some failures are due to yes/no questions where LLM gives verbose answers

## Demo (Quick Test)

Test the basic RAG functionality without the full evaluation dataset:

1. Start server: `uvicorn main:app --reload`
2. Open http://127.0.0.1:8000
3. Ask: "Top 1 ranking team in year 2003" → Should give wrong answer
4. Upload `sample/ranking-2003.txt`
5. Ask again → Should now give correct answer

## License

MIT
