import streamlit as st
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import re

# Load .env
load_dotenv()

# Streamlit config
st.set_page_config(page_title="YouTube Video Q&A", layout="centered")
st.title("🎥 YouTube Video Chatbot using RAG + Groq + Huggingface")

# Inputs
video_url = st.text_input("Paste a YouTube video URL 👇")
question = st.text_input("Ask a question about the video 🎯")

# YouTube ID extractor
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# Transcript + Vector
@st.cache_data(show_spinner="📄 Fetching and processing transcript...")
def process_video(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        full_text = " ".join(chunk["text"] for chunk in transcript)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([full_text])

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(docs, embeddings)

        return vectorstore
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

# Chat history store
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# If URL + question provided
if video_url and question:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
    else:
        vectorstore = process_video(video_id)
        if vectorstore is None:
            st.warning("⚠️ No transcript available.")
        else:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

            # Helper to format docs
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Prompt that expects history_messages
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a helpful assistant that only answers questions using the transcript of a YouTube video. "
                 "If the answer is not in the transcript, say: 'I don't know based on the transcript.'\n\n"
                 "Transcript:\n{context}"),
                MessagesPlaceholder(variable_name="history_messages"),
                ("human", "{question}")
            ])

            # Chain step: first fetch context
            def retrieve_and_format(inputs):
                docs = retriever.invoke(inputs["question"])
                return {
                    "context": format_docs(docs),
                    "question": inputs["question"]
                }

            rag_chain = RunnableLambda(retrieve_and_format)

            # Wrap the PROMPT with history handler
            prompt_with_history = RunnableWithMessageHistory(
                prompt,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history_messages"
            )

            # Full pipeline
            full_chain = rag_chain | prompt_with_history | llm | StrOutputParser()

            with st.spinner("💬 Generating answer..."):
                try:
                    response = full_chain.invoke(
                        {"question": question},
                        config={"configurable": {"session_id": video_id}}
                    )
                    st.success("✅ Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"⚠️ Error during generation: {e}")
