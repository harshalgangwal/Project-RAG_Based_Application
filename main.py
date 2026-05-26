import streamlit as st
import pickle
import time
from PIL import Image
import os
import warnings

warnings.filterwarnings("ignore")

from transformers import pipeline

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI News Research Assistant",
    page_icon="📰",
    layout="wide"
)


# ======================================
# CUSTOM CSS
# ======================================
st.markdown("""
<style>

.main {
    background-color: #020617;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

.stTextInput input {
    border-radius: 12px;
    border: 1px solid #334155;
    background-color: #1e293b;
    color: white;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 18px;
    border: none;
}

h1, h2, h3 {
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)


# ======================================
# LOAD VECTOR DATABASE
# ======================================
def load_faiss_index():

    faiss_index_path = "faiss_index"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_index = FAISS.load_local(
        faiss_index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_index


# ======================================
# LOAD LLM
# ======================================
@st.cache_resource
def load_llm_pipeline():

    model_id = "distilgpt2"

    pipe = pipeline(
        task="text-generation",
        model=model_id,
        device=-1,
        max_new_tokens=150
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# ======================================
# MAIN APPLICATION
# ======================================
def main():

    # HEADER
    col1, col2 = st.columns([8, 1])

    with col1:
        st.title("📰 AI News Research Assistant")
        st.markdown(
            "Analyze news articles using **RAG (Retrieval-Augmented Generation)**"
        )

    with col2:
        if os.path.exists("image.png"):
            image = Image.open("image.png")
            st.image(image, width=100)

    st.markdown("---")

    # SIDEBAR
    st.sidebar.title("🔗 News Article URLs")

    urls = []

    for i in range(2):
        url = st.sidebar.text_input(f"Article URL {i+1}")
        if url:
            urls.append(url)

    process_url_clicked = st.sidebar.button("🚀 Process URLs")

    faiss_index_path = "faiss_index"

    status_placeholder = st.empty()

    # ======================================
    # PROCESS URLS
    # ======================================
    if process_url_clicked:

        if not urls:
            st.warning("⚠ Please enter at least one URL")
            return

        try:

            # LOAD DOCUMENTS
            status_placeholder.info("📥 Loading articles...")

            loader = WebBaseLoader(urls)

            data = loader.load()

            # SPLIT TEXT
            status_placeholder.info("✂ Splitting text into chunks...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(data)

            # EMBEDDINGS
            status_placeholder.info("🧠 Creating embeddings...")

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # CREATE VECTOR STORE
            vector_index = FAISS.from_documents(
                docs,
                embedding_model
            )

            # SAVE VECTOR STORE
            vector_index.save_local(faiss_index_path)

            # SAVE DOCS
            with open(f"{faiss_index_path}/metadata.pkl", "wb") as f:
                pickle.dump(docs, f)

            time.sleep(1)

            status_placeholder.success("✅ Articles processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing URLs: {e}")

    # ======================================
    # ASK QUESTION
    # ======================================
    st.markdown("## 💬 Ask Questions")

    query = st.text_input(
        "Enter your question about the articles"
    )

    # ======================================
    # ANSWER QUESTION
    # ======================================
    if query:

        if os.path.exists(faiss_index_path):

            try:

                with st.spinner("🤖 AI is analyzing articles..."):

                    vector_index = load_faiss_index()

                    llm = load_llm_pipeline()

                    chain = RetrievalQA.from_llm(
                        llm=llm,
                        retriever=vector_index.as_retriever()
                    )

                    docs = vector_index.similarity_search(query)

                    if not docs:
                        st.warning("⚠ No relevant information found.")
                    else:

                        result = chain.invoke({"query": query})

                        st.markdown("## ✅ Answer")

                        st.success(result["result"])

                        st.markdown("## 📚 Retrieved Sources")

                        for i, doc in enumerate(docs[:3]):

                            with st.expander(f"Source Chunk {i+1}"):

                                st.write(doc.page_content)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

        else:
            st.warning("⚠ Please process URLs first.")


# ======================================
# RUN APP
# ======================================
if __name__ == "__main__":
    main()
