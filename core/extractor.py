import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------
# LLM
# -----------------------------------

llm = ChatMistralAI(
    model="open-mistral-7b",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)


# -----------------------------------
# Text Splitter
# -----------------------------------

def split_transcript(transcript: str) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


# -----------------------------------
# Generic Chain Builder
# -----------------------------------

def build_chain(system_prompt: str):

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    return prompt | llm | StrOutputParser()


# -----------------------------------
# Generic Chunk Processor
# -----------------------------------

def process_chunks(transcript: str, system_prompt: str) -> str:

    chain = build_chain(system_prompt)

    chunks = split_transcript(transcript)

    results = chain.batch([
        {"text": chunk}
        for chunk in chunks
    ])

    return "\n\n".join(results)


# -----------------------------------
# Action Items
# -----------------------------------

ACTION_ITEMS_PROMPT = """
You are an expert meeting analyst.

From the meeting transcript, extract all action items.

For each provide:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else write 'Not specified')

Format as a numbered list.

If none found, say:
'No action items found.'
"""


def extract_action_items(transcript: str) -> str:

    return process_chunks(
        transcript,
        ACTION_ITEMS_PROMPT
    )


# -----------------------------------
# Key Decisions
# -----------------------------------

KEY_DECISIONS_PROMPT = """
You are an expert meeting analyst.

From the meeting transcript, extract all key decisions made.

Format as a numbered list.

If none found, say:
'No key decisions found.'
"""


def extract_key_decisions(transcript: str) -> str:

    return process_chunks(
        transcript,
        KEY_DECISIONS_PROMPT
    )


# -----------------------------------
# Open Questions
# -----------------------------------

QUESTIONS_PROMPT = """
You are an expert meeting analyst.

From the meeting transcript, extract all unresolved questions
or topics needing follow-up.

Format as a numbered list.

If none found, say:
'No open questions found.'
"""


def extract_questions(transcript: str) -> str:

    return process_chunks(
        transcript,
        QUESTIONS_PROMPT
    )