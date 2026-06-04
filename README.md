# Literature-RAG: Document-Bounded Research Discovery Engine

An academic-focused Retrieval-Augmented Generation (RAG) system engineered to parse, index, and reason over dense scientific literature. Unlike traditional RAG frameworks that naively split texts by fixed token counts—disrupting architectural descriptions and equations—this pipeline enforces structural document boundaries to preserve absolute contextual continuity.

### Core Architecture & Workflow
1. **Structural Ingestion:** Parses dense extraction text files and separates papers cleanly using absolute semantic markers to ensure titles, methodologies, and abstracts remain in unified blocks.
2. **Vector Space Mapping:** Computes high-density semantic vector representations via `all-MiniLM-L6-v2` SentenceTransformers.
3. **L2-Normalized Dense Indexing:** Utilizes a FAISS `IndexFlatIP` (Inner Product) index with normalized embeddings to perform exact cosine similarity matching during the retrieval stage.
4. **Context-Constrained Inference:** Routes targeted context chunks through Groq's high-throughput `llama-3.1-8b-instant` API, maintaining an active, pruned conversational memory buffer for multi-turn literature synthesis.

### Tech Stack
* **Language Model:** Llama-3.1-8b (via Groq Cloud SDK)
* **Embeddings Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Core Libraries:** NumPy, Python 3.x
