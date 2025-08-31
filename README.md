# College-admission-Enquiry-Chatbot (RAG + LLM)

🎓 An intelligent Conversational AI system designed to handle college admission-related queries such as admissions, fees, deadlines, and programs. This chatbot leverages Retrieval-Augmented Generation (RAG), combining semantic search with Large Language Models (LLMs) for accurate, context-aware, and human-like responses.

🔑 Key Features

- 🧠 Retrieval-Augmented Generation (RAG):
Combines Sentence-BERT embeddings with FAISS vector search to fetch relevant information from a custom admissions dataset.
- 🤖 LLM-Powered Responses:
Uses Falcon-7B-Instruct (Hugging Face) to generate natural and context-grounded answers.
- 📚 Custom Knowledge Base:
Admission-specific FAQ dataset (Chatbot_data.csv) powers domain knowledge.

🧠 Technical Stack

- Python
- Sentence-BERT (all-MiniLM-L6-v2) → Embedding generation
- FAISS → Fast similarity search
- Falcon-7B-Instruct → Generative LLM for natural answers
- Transformers, Accelerate, Torch → Model loading & inference
