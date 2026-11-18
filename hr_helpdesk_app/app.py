
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Zyro Dynamics")
st.subheader("AI-Powered HR Help Desk")
st.caption("Your intelligent assistant for HR policies and employee support.")

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Please configure GROQ_API_KEY in Streamlit Secrets.")
        st.stop()


# LOAD HR DOCUMENTS
@st.cache_resource
def load_hr_system():

    pdf_dir = os.path.join(
        os.path.dirname(__file__),
        "hr_policies"
    )

    documents = []

    for filename in sorted(os.listdir(pdf_dir)):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join(pdf_dir, filename)
            )
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024
    )

    return retriever, llm


with st.spinner("Loading HR policies and AI system..."):
    retriever, llm = load_hr_system()

st.success("HR Help Desk is ready!")


# PROMPTS
relevance_prompt = ChatPromptTemplate.from_template("""
You are an HR relevance checker.

Determine whether the provided HR policy context contains
information that can answer the employee's question.

Return only YES or NO.

Question:
{question}

HR Context:
{context}
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are an intelligent HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Answer using ONLY the provided HR policy context.

Rules:
1. Give accurate and simple answers.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is unavailable, say:
"I could not find this information in the available HR policies.
Please contact the HR department."

HR Context:
{context}

Employee Question:
{question}

Answer:
""")


# CHATBOT FUNCTION
def ask_hr(question):

    docs = retriever.invoke(question)

    if not docs:
        return (
            "I could not find relevant information in the available "
            "HR policies. Please contact the HR department."
        ), []

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    relevance_result = llm.invoke(
        relevance_prompt.format_messages(
            question=question,
            context=context
        )
    )

    if relevance_result.content.strip().upper() != "YES":
        return (
            "I could not find relevant information in the available "
            "HR policies. Please contact the HR department."
        ), []

    answer_result = llm.invoke(
        answer_prompt.format_messages(
            question=question,
            context=context
        )
    )

    return answer_result.content, docs


# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View HR Policy Sources"):
                for source in message["sources"]:
                    st.write(
                        f"📄 {source['file']} | Page {source['page']}"
                    )


# CHAT INPUT
question = st.chat_input("Ask anything about HR policies...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            try:
                answer, docs = ask_hr(question)

                sources = [
                    {
                        "file": os.path.basename(
                            doc.metadata.get("source", "HR Policy")
                        ),
                        "page": doc.metadata.get("page", "N/A")
                    }
                    for doc in docs
                ]

                st.markdown(answer)

                if sources:
                    with st.expander("View HR Policy Sources"):
                        for source in sources:
                            st.write(
                                f"📄 {source['file']} | Page {source['page']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.title("About")
st.sidebar.write(
    "AI-powered HR assistant using Retrieval-Augmented Generation (RAG)."
)
st.sidebar.write("Powered by Groq, FAISS and Sentence Transformers.")
