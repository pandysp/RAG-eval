# Vibe Coding Meetup Notes

Meta-learnings from a hackathon session building a RAG evaluation system.

---

## 0. Background: How We Found the Starting Point

**The prompt to ChatGPT:**
> "Please give me a GitHub repo with a very simple API based RAG system that I can run locally with very little effort. It should already do ingestion and chunking and stuff."

**ChatGPT's recommendation:**
- Repo: `ChocoPancakes1219/RAG-application-using-LlamaIndex`
- Why: Small, explicit POST /ingest and GET /query, stores index locally, README walkthrough
- Gotcha noted: Requires OpenAI API key (not fully offline)

**Key insight**: Used one AI (ChatGPT) to find a starting point, then another AI (Claude Code) to extend it. Multi-AI workflow - ChatGPT is excellent at finding stuff online (web search, repo discovery), Claude Code excels at implementation.

---

## 1. Setup Summary

| Component | Choice |
|-----------|--------|
| **Coding Model** | Claude Opus 4.5 |
| **Coding Harness** | Claude Code CLI |
| **Skills/Plugins** | Built-in only (no MCP servers) |
| **CLAUDE.md** | None configured |

---

## 2. Meta-Learnings: How We Approached the Problem

### User Coordination Techniques

- **Iterative scoping**: Started with 1001 PDFs → hit time wall → scoped down to 20 PDFs (200 questions)
- **Decision steering**: User rejected full LLM-as-judge → proposed hybrid (LLM extracts keywords once, fast string match for eval)
- **Plan-then-execute**: User requested plan mode before implementation
- **Verification request**: User asked to "follow the README" to catch errors

### AI Collaboration Patterns

- **Background tasks**: Long-running indexing in background while discussing next steps
- **Parallel exploration**: Multiple operations at once (check status + read files)
- **Course correction**: User interrupted slow approaches ("this will take 45 min" → "let's subset")

---

## 3. Time Breakdown

**Context**: Hackathon time budget started at 30 min, extended to ~1:30h

| Phase | Duration | What Happened |
|-------|----------|---------------|
| Setup & debugging | ~15 min | Python 3.13 compatibility, requirements.txt fixes |
| Data exploration | ~10 min | 1001 PDFs → realized too slow → subset to 200 questions |
| Indexing pivot | ~5 min | Switched OpenAI embeddings → local BGE model (50s vs 45min) |
| Plan discussion | ~10 min | Evaluated LLM-as-judge vs keyword matching |
| Implementation | ~15 min | Created evaluate.py, extract_keywords.py, modified main.py |
| Documentation | ~10 min | README rewrite, repo cleanup |
| Verification | ~5 min | Actually followed README, caught 3 errors |

**Total**: ~70 min of active work

---

## 4. Alternative Approaches Considered

| Approach | Why Rejected |
|----------|--------------|
| Full 1001 PDF dataset | Too slow for hackathon (~45 min indexing) |
| OpenAI embeddings | API rate limits, cost, slow |
| Pure LLM-as-judge eval | Too slow ($$$) for 200 questions |
| Simple substring matching | Too brittle ("The answer is X" ≠ "X") |

**Final choice**: Hybrid approach - LLM extracts keywords once (~$0.10), fast string matching for evaluation

---

## 5. Closed-Loop Verification Patterns

**Key insight**: Create feedback loops so the LLM can verify its own work.

| Pattern | How We Used It | What It Caught |
|---------|----------------|----------------|
| **Install packages yourself** | Ran `pip install` → hit Python 3.13 errors | Fixed numpy, greenlet compatibility |
| **Start the server yourself** | Ran `uvicorn main:app` → tested endpoints | Confirmed `/query_with_context` worked |
| **Follow your own README** | Walked through setup steps literally | Wrong unzip command, missing storage note, keywords pre-extracted |
| **Run the eval yourself** | Executed full pipeline before documenting | Verified 98% hit rate, 62% correctness |
| **Check the browser** | Opened localhost:8000 | Confirmed UI accessible |

### Why This Matters

LLMs can write plausible-looking docs/code that fails on execution. Closed loops catch these errors before the user does.

### Examples of Closed Loops to Create

- **TDD**: Write test → watch it fail → implement → watch it pass
- **README verification**: Follow every step as a new user would
- **API testing**: `curl` endpoints after implementing
- **UI testing**: Actually open the browser and click around
- **Build verification**: Run the build/deploy pipeline

---

## 6. Key Takeaways for Vibe Coders

1. **Scope to your time budget** - Hackathon was 30min→1:30h; 20 PDFs proved the concept without burning the clock

2. **Plan mode pays off** - Upfront design prevented rework on non-trivial tasks

3. **Create closed loops** - Let the LLM verify its own work (install, run, test, follow docs)

4. **Multi-AI workflow** - ChatGPT excels at finding stuff online; Claude Code excels at implementation

5. **"YOLO mode" can surprise you** - Sometimes don't specify everything; let the agent find creative solutions
   - Example: Agent proposed keyword extraction hybrid approach (better than pure LLM-as-judge)
   - Caveat: Not always better - user steering still essential for course correction
