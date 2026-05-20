import os

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an expert meeting assistant.

Answer the user's question ONLY using the provided meeting transcript context.

Rules:
- Do not hallucinate
- If answer is unavailable, say:
  "I could not find this information in the meeting transcript."
- Keep answers concise and precise
- Clearly mention speaker names if referenced
- Do not make assumptions

Meeting Transcript Context:
{context}
"""


# =========================================================
# PROMPT TEMPLATE
# =========================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ]
)


# =========================================================
# LOAD LLM
# =========================================================

def get_llm() -> ChatMistralAI:

    return ChatMistralAI(
        model="open-mistral-7b",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


# =========================================================
# FORMAT RETRIEVED DOCS
# =========================================================

def format_docs(docs) -> str:

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# =========================================================
# BUILD RAG CHAIN
# =========================================================

def build_rag_chain(transcript: str):

    print("\n[INFO] Building RAG Chain...")

    # -------------------------
    # BUILD VECTOR STORE
    # -------------------------

    vector_store = build_vector_store(transcript)

    # -------------------------
    # RETRIEVER
    # -------------------------

    retriever = get_retriever(
        vector_store=vector_store,
        k=4
    )

    # -------------------------
    # LLM
    # -------------------------

    llm = get_llm()

    # -------------------------
    # LCEL RAG PIPELINE
    # -------------------------

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    print("[INFO] RAG Chain Built Successfully!")

    return rag_chain


# =========================================================
# LOAD EXISTING RAG CHAIN
# =========================================================

def load_rag_chain():

    print("\n[INFO] Loading Existing RAG Chain...")
   
    # LOAD VECTOR STORE
    vector_store = load_vector_store()

    # RETRIEVER
    retriever = get_retriever(
        vector_store=vector_store,
        k=4
    )
    # LLM
    llm = get_llm()
    
    # LCEL RAG PIPELINE
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    print("[INFO] Existing RAG Chain Loaded!")
    return rag_chain


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(
    rag_chain,
    question: str
) -> str:

    print(f"\n[QUESTION] {question}")

    answer = rag_chain.invoke(question)

    print(f"\n[ANSWER]\n{answer}")

    return answer