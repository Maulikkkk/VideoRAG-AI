from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
load_dotenv()

from core.summarize import (
    summarize,
    generate_title
)
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import (
    build_rag_chain,
    ask_question
)

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(
    source: str,
    language: str = "english"
) -> dict:

    print("\n" + "=" * 60)
    print("🚀 Starting AI Video Assistant Pipeline")
    print("=" * 60)

    # =====================================================
    # STEP 1 — PROCESS INPUT
    # =====================================================

    print("\n[STEP 1] Processing Input Source...")

    chunks = process_input(source)

    print(f"[INFO] Total Audio Chunks: {len(chunks)}")

    # =====================================================
    # STEP 2 — TRANSCRIPTION
    # =====================================================

    print("\n[STEP 2] Transcribing Audio...")

    transcript = transcribe_all(
        chunks,
        language
    )

    print(
        f"\n[INFO] Transcript Preview:\n"
        f"{transcript[:300]}..."
    )

    # =====================================================
    # STEP 3 — TITLE GENERATION
    # =====================================================

    print("\n[STEP 3] Generating Title...")

    title = generate_title(transcript)

    # =====================================================
    # STEP 4 — SUMMARIZATION
    # =====================================================

    print("\n[STEP 4] Generating Summary...")

    summary = summarize(transcript)

    # =====================================================
    # STEP 5 — EXTRACTION TASKS
    # =====================================================

    print("\n[STEP 5] Extracting Insights...")

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    # =====================================================
    # STEP 6 — BUILD RAG CHAIN
    # =====================================================

    print("\n[STEP 6] Building RAG Pipeline...")

    rag_chain = build_rag_chain(transcript)

    print("\n✅ Pipeline Completed Successfully!")

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(result: dict):

    print("\n" + "=" * 60)

    print(f"\n📌 TITLE:\n{result['title']}")

    print(f"\n📋 SUMMARY:\n{result['summary']}")

    print(f"\n✅ ACTION ITEMS:\n{result['action_items']}")

    print(f"\n🔑 KEY DECISIONS:\n{result['key_decisions']}")

    print(f"\n❓ OPEN QUESTIONS:\n{result['open_questions']}")

    print("\n" + "=" * 60)


# =========================================================
# CHAT LOOP
# =========================================================

def start_chat_session(rag_chain):

    print("\n💬 Chat with Your Video")
    print("Type 'exit' anytime to quit.\n")

    while True:

        question = input("You: ").strip()

        # -----------------------------
        # EXIT CONDITIONS
        # -----------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:
            print("\n👋 Goodbye!")
            break

        # -----------------------------
        # EMPTY INPUT
        # -----------------------------

        if not question:
            continue

        # -----------------------------
        # ASK QUESTION
        # -----------------------------

        answer = ask_question(
            rag_chain,
            question
        )

        print(f"\n🤖 Assistant: {answer}\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    print("\n🎥 AI VIDEO ASSISTANT\n")

    source = input(
        "Enter YouTube URL or Local File Path:\n> "
    ).strip()

    language = input(
        "\nLanguage (english / hinglish):\n> "
    ).strip().lower()

    if not language:
        language = "english"

    # =====================================================
    # RUN COMPLETE PIPELINE
    # =====================================================

    result = run_pipeline(
        source=source,
        language=language
    )

    # =====================================================
    # DISPLAY OUTPUTS
    # =====================================================

    display_results(result)

    # =====================================================
    # START RAG CHAT
    # =====================================================

    start_chat_session(
        result["rag_chain"]
    )