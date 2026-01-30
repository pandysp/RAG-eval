# RAG Application with Evaluation Framework

A Retrieval-Augmented Generation (RAG) web application built with LlamaIndex and FastAPI, including a baseline evaluation framework for measuring system quality.

## Overview

This application enhances a large language model with RAG capabilities, allowing it to reference an external knowledge base for more accurate and contextually relevant responses. It includes:

- **RAG Web Application**: Upload documents, query them via a chat interface
- **Evaluation Framework**: Measure retrieval and answer quality using keyword-based metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r Requirements.txt
```

### 2. Configure API Key

```bash
cp openai_key.env.example openai_key.env
# Edit openai_key.env and add your OpenAI API key
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Access the web interface at http://127.0.0.1:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface for document upload and querying |
| `/ingest` | POST | Upload and index documents (PDF, TXT) |
| `/query` | GET | Query the knowledge base |
| `/query_with_context` | GET | Query with source documents (for evaluation) |

## Evaluation Framework

The evaluation system measures RAG quality using keyword-based matching - a fast, deterministic approach that doesn't require LLM calls during evaluation.

### Metrics

| Metric | Description |
|--------|-------------|
| **Retrieval Hit Rate** | % of queries where the correct source PDF appears in retrieved chunks |
| **Keyword Match Score** | % of expected keywords found in the generated answer |
| **Answer Correctness** | % of answers with ≥50% keyword match |

### Running Evaluation

#### Step 1: Extract Keywords (One-time)

```bash
python extract_keywords.py
```

This uses GPT-4o-mini to extract 2-5 key terms from each expected answer in `Eval_data_subset.csv`.

#### Step 2: Run Evaluation

```bash
# Make sure the server is running
uvicorn main:app &

# Run evaluation
python evaluate.py
```

### Baseline Results

| Metric | Value |
|--------|-------|
| Questions Evaluated | 200 |
| Retrieval Hit Rate | 98.0% |
| Avg Keyword Match | 58.7% |
| Answer Correctness | 62.0% |

Results are saved to `eval_results.csv` with per-question details.

## Project Structure

```
├── main.py                  # FastAPI application
├── chat_interface.html      # Web UI
├── evaluate.py              # Evaluation script
├── extract_keywords.py      # Keyword extraction (one-time)
├── Eval_data_subset.csv     # 200 evaluation questions with keywords
├── eval_results.csv         # Evaluation results
├── Requirements.txt         # Python dependencies
├── openai_key.env.example   # API key template
├── storage/                 # Vector index storage
├── data/                    # Sample documents
└── sample/                  # Test files for demo
```

## How It Works

### RAG Pipeline

1. **Document Ingestion**: Documents are chunked (512 tokens, 50 overlap) and embedded using `BAAI/bge-small-en-v1.5`
2. **Index Storage**: Vectors stored locally in `storage/` directory
3. **Query Processing**: Queries are embedded, similar chunks retrieved, and response generated via tree summarization

### Evaluation Approach

1. **Keyword Extraction**: LLM extracts 2-5 core concepts from each expected answer
2. **Fast Matching**: During evaluation, check if keywords appear in generated answers (string matching, no LLM calls)
3. **Retrieval Verification**: Check if source PDF filename appears in retrieved chunks

## Configuration

Key settings in `main.py`:

```python
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

## Demo

1. Ask "Top 1 ranking team in year 2003" - should give incorrect answer (not in dataset)
2. Upload `sample/ranking-2003.txt` via the web interface
3. Ask again - should now give correct answer
4. Check that no files remain in `data/` folder (temp files cleaned up)

## License

MIT
