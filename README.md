<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/Mistral_AI-F7931E?style=for-the-badge&logo=mistral&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22d3ee?style=for-the-badge"/>

<br/><br/>

```
██╗   ██╗██╗██████╗ ███████╗ ██████╗ ██████╗  █████╗  ██████╗      █████╗ ██╗
██║   ██║██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝     ██╔══██╗██║
██║   ██║██║██║  ██║█████╗  ██║   ██║██████╔╝███████║██║  ███╗    ███████║██║
╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║   ██║    ██╔══██║██║
 ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝██║  ██║██║  ██║╚██████╔╝    ██║  ██║██║
  ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝
```

### AI-Powered Video Understanding & Conversational RAG

**Drop a YouTube link. Get a transcript, summary, action items, and a full chat interface — in minutes.**

<br/>

[**View Demo**](#-demo) · [**Quick Start**](#-installation) · [**Architecture**](#️-architecture) · [**Features**](#-features)

<br/>

</div>

---

## 🧠 What is VideoRAG-AI?

VideoRAG-AI is an end-to-end AI assistant that turns any video — YouTube, meeting recording, podcast, or lecture — into a fully searchable, conversational knowledge base.

No more scrubbing through hours of content. Ask questions, get answers, grounded in what was actually said.

```
YouTube URL  ──→  Audio  ──→  Transcript  ──→  Summary + Action Items
                                    ↓
                              Vector Store
                                    ↓
                           Chat with your video 💬
```

---

## ✨ Features

| Capability | Details |
|---|---|
| 📥 **Multi-Source Input** | YouTube URLs, local MP4/audio files, meeting recordings |
| 🗣️ **Multilingual ASR** | English via Whisper · Hindi/Hinglish via Sarvam AI |
| 📋 **AI Summarization** | Map-reduce pipeline for long-form content |
| ✅ **Action Item Extraction** | Automatically surface tasks from any meeting |
| 🔑 **Key Decision Detection** | Identify what was decided, without re-watching |
| ❓ **Open Question Mining** | Flag unresolved items from conversations |
| 🔍 **Semantic RAG Search** | ChromaDB + HuggingFace embeddings + Mistral LLM |
| 💬 **Conversational Chat** | Full chat UI to interrogate any video transcript |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VideoRAG-AI Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Video URL / File                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│   │ Audio       │───▶│ Chunking     │───▶│ Speech-to-Text    │  │
│   │ Extraction  │    │ (pydub)      │    │ Whisper / Sarvam  │  │
│   │ (yt-dlp)    │    └──────────────┘    └────────┬──────────┘  │
│   └─────────────┘                                 │             │
│                                                   ▼             │
│                                         Full Transcript         │
│                                                   │             │
│                          ┌────────────────────────┤             │
│                          │                        │             │
│                          ▼                        ▼             │
│                  ┌───────────────┐      ┌──────────────────┐   │
│                  │ AI Analysis   │      │ Vector Pipeline  │   │
│                  │ • Summary     │      │ • Text Chunks    │   │
│                  │ • Actions     │      │ • Embeddings     │   │
│                  │ • Decisions   │      │ • ChromaDB Store │   │
│                  │ • Questions   │      └────────┬─────────┘   │
│                  └───────────────┘               │             │
│                                                  ▼             │
│                                          RAG Retriever          │
│                                                  │             │
│                                                  ▼             │
│                                       Mistral LLM Response      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

<table>
<tr>
<td>

**AI / LLM**
- Mistral AI
- LangChain
- RAG (Retrieval-Augmented Generation)

**Speech-to-Text**
- OpenAI Whisper *(English)*
- Sarvam AI *(Hindi/Hinglish)*

</td>
<td>

**Vector Store**
- ChromaDB
- HuggingFace Embeddings
- `sentence-transformers/all-MiniLM-L6-v2`

**Audio / Video**
- yt-dlp
- ffmpeg
- pydub

</td>
<td>

**Frontend**
- Streamlit

**Backend**
- Python 3.11

**Environment**
- python-dotenv
- Virtual environment

</td>
</tr>
</table>

---

## 📂 Project Structure

```
VideoRAG-AI/
│
├── app.py                  # Streamlit UI entrypoint
├── main.py                 # CLI entrypoint
├── requirements.txt
├── .env                    # API keys (not committed)
│
├── core/
│   ├── transcriber.py      # Whisper + Sarvam ASR
│   ├── summarizer.py       # Summarization + title generation
│   ├── extractor.py        # Action items, decisions, questions
│   ├── rag_engine.py       # RAG chain + conversational QA
│   └── vector_store.py     # Embeddings + ChromaDB
│
├── utils/
│   └── audio_processor.py  # yt-dlp + ffmpeg + chunking
│
└── vector_db/              # Persistent ChromaDB storage
```

---

## 🚀 Installation

### 1. Clone

```bash
git clone https://github.com/Maulikkkk/VideoRAG-AI.git
cd VideoRAG-AI
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_key
SARVAM_API_KEY=your_sarvam_key
WHISPER_MODEL=base
SARVAM_STT_MODEL=saaras:v2.5
```

> Get your Mistral key at [console.mistral.ai](https://console.mistral.ai) · Sarvam key at [sarvam.ai](https://sarvam.ai)

### 5. Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 💬 Usage

1. Paste a **YouTube URL** or local file path into the sidebar
2. Select **language** (English or Hinglish)
3. Hit **Run Analysis** and watch the pipeline execute live
4. Review the **summary, action items, decisions, and open questions**
5. Use the **chat interface** to ask anything about the video

---

## 🗺️ Roadmap

- [ ] Streaming LLM responses
- [ ] Speaker diarization (who said what)
- [ ] Real-time meeting assistant mode
- [ ] Multi-modal retrieval (video frames + text)
- [ ] Memory-enabled multi-turn RAG
- [ ] GPU acceleration for Whisper
- [ ] LangGraph agentic workflows
- [ ] Export to PDF / Notion

---

## 📸 Demo

> *Screenshots and demo GIF coming soon.*

---

## 🤝 Contributing

Contributions are welcome. Open an issue first to discuss what you'd like to change.

---

## 📜 License

[MIT](LICENSE)

---

<div align="center">

Built by **Maulik Gupta** · AI/ML Engineer · RAG & LLM Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Maulikkkk-181717?style=flat-square&logo=github)](https://github.com/Maulikkkk)

</div>
