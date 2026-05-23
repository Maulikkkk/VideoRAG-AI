import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
# ----------------------------
# LLM
# ----------------------------

def get_llm():
    return ChatMistralAI(
        model="open-mistral-7b",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


# ----------------------------
# Text Splitter
# ----------------------------

def split_transcript(transcript: str) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


# ----------------------------
# Prompts
# ----------------------------

MAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize this portion of a meeting transcript concisely.",
    ),
    ("human", "{text}"),
])


COMBINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert meeting summarizer. "
        "Combine these partial summaries into one final professional "
        "meeting summary in bullet points.",
    ),
    ("human", "{text}"),
])


TITLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Based on the meeting transcript, generate a short "
        "professional meeting title (max 8 words). "
        "Only return the title.",
    ),
    ("human", "{text}"),
])


# ----------------------------
# Summarization
# ----------------------------

def summarize(transcript: str) -> str:

    llm = get_llm()

    map_chain = MAP_PROMPT | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined_summary = "\n\n".join(chunk_summaries)

    combine_chain = COMBINE_PROMPT | llm | StrOutputParser()

    return combine_chain.invoke({
        "text": combined_summary
    })


# ----------------------------
# Title Generation
# ----------------------------

def generate_title(transcript: str) -> str:

    llm = get_llm()

    title_chain = TITLE_PROMPT | llm | StrOutputParser()

    return title_chain.invoke({
        "text": transcript[:2000]
    })