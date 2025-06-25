# 🎥 YouTube Video Chatbot using RAG + LangChain + Groq + HuggingFace

This is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to:

- 🔗 Paste any **YouTube video URL**
- ❓ Ask **any question** based on the video transcript
- 💬 Get context-aware answers — with **chat history memory**!

Built using **LangChain**, **Groq LLaMA-3**, and **HuggingFace embeddings**, and served through a simple **Streamlit UI**.

---

## 🚀 Features

✅ Ask questions about any YouTube video  
✅ Automatically fetches and indexes the transcript  
✅ Uses **similarity search (FAISS)** for relevant content  
✅ **Chat history is remembered** with `RunnableWithMessageHistory`  
✅ Answers only from transcript – avoids hallucination  
✅ Built with modular LangChain components  
✅ Fast, responsive UI with Streamlit

---

## 🧠 Tech Stack

| Tool                           | Purpose                         |
| ------------------------------ | ------------------------------- |
| **Streamlit**                  | Web interface                   |
| **LangChain**                  | RAG + Memory handling           |
| **Groq API**                   | Fast LLM (LLaMA 3.1 8B Instant) |
| **FAISS**                      | Vector similarity search        |
| **HuggingFace Transformers**   | Embeddings (`all-MiniLM-L6-v2`) |
| **YouTubeTranscriptAPI**       | Transcript extraction           |
| **RunnableWithMessageHistory** | Persistent chat memory          |

---

## 📁 Folder Structure

        📦 youtube-video-rag-chatbot/
        ├── app.py # Main Streamlit app
        ├── requirements.txt # Python dependencies
        ├── .env # API keys
        └── README.md # You're here!

Install dependencies

pip install -r requirements.txt

streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-huggingface
youtube-transcript-api
faiss-cpu
sentence-transformers
python-dotenv

Configure your API keys
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=HUGGINGFACEHUB_API_TOKEN

▶️ Run the App

streamlit run app.py
Visit http://localhost:8501 in your browser.

💬 How Chat History Works
This app uses LangChain's RunnableWithMessageHistory:

Each session stores user questions and LLM responses.

Memory is scoped by a session_id (automatically set to video ID).

Users can ask follow-up questions, and the model remembers previous turns.

📌 Limitations
Only works for videos with English transcripts available.

Memory is per-session (not persistent across browser refresh unless you extend it).

Currently supports only public YouTube videos.

📬 Credits
Created with ❤️ using:

LangChain

Groq

Streamlit

HuggingFace Embeddings
