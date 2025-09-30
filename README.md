<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">


# LUNA-AI

<em>Empowering Seamless Conversations with Intelligent Voice</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/Djninja926/luna-ai?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/Djninja926/luna-ai?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/Djninja926/luna-ai?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Ollama-000000.svg?style=flat&logo=Ollama&logoColor=white" alt="Ollama">
<img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=flat&logo=LangChain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white" alt="OpenAI">

</div>
<br>

---

## Overview

Luna-ai is an innovative developer toolkit that enables the creation of intelligent, voice-driven applications with real-time interaction and contextual understanding. It orchestrates speech recognition, natural language processing, and AI response generation into a cohesive system, simplifying the development of hands-free, conversational interfaces.

**Why Luna-ai?**

This project empowers developers to build seamless voice interfaces that are both responsive and contextually aware. The core features include:

- 🧠 **Microphone & Wake Word Detection:** Hands-free, real-time voice activation for effortless user engagement.
- 🎙️ **Speech Transcription:** Converts spoken language into text instantly, supporting natural interactions.
- 🔗 **Contextual AI Conversations:** Maintains persistent, searchable conversation history for coherent dialogues.
- ⚙️ **Modular Architecture:** Ensures reliable data flow and system cohesion across components.
- 🚀 **Scalable Integration:** Combines advanced speech recognition, NLP, and AI models for robust applications.

---

## Features

|      | Component       | Details                                                                                     |
| :--- | :-------------- | :------------------------------------------------------------------------------------------ |
| ⚙️  | **Architecture**  | <ul><li>Modular design separating core functionalities (e.g., model handling, data processing)</li><li>Supports multiple AI models and pipelines</li><li>Likely employs layered architecture for scalability</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Uses type hints (`typing`, `pydantic`) for robustness</li><li>Follows PEP8 standards, with organized directory structure</li><li>Includes custom modules for specific tasks (e.g., `langchain`, `transformers` integration)</li></ul> |
| 📄 | **Documentation** | <ul><li>README provides project overview, setup instructions, dependencies</li><li>Likely includes inline docstrings and possibly API docs</li><li>Uses markdown for clarity, references external docs for models and APIs</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Extensive use of third-party libraries: `transformers`, `langchain`, `onnxruntime`, `ctranslate2`</li><li>Supports cloud/AI services like `ollama`, `pinecone`, `openai`</li><li>Integrates audio processing (`PyAudio`, `soundfile`, `whisper`), ML frameworks (`torch`, `scikit-learn`)</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Component-based design with separate modules for models, data, and APIs</li><li>Configurable via environment variables and config files</li><li>Supports plugin-like extensions (e.g., different backends, models)</li></ul> |
| 🧪 | **Testing**       | <ul><li>Likely includes unit tests for core modules</li><li>Uses testing frameworks compatible with Python (`pytest` or similar)</li><li>Test coverage inferred from dependencies and code structure</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Utilizes optimized libraries (`onnxruntime`, `librosa`, `numba`) for speed</li><li>Supports hardware acceleration (GPU via `torch`, `onnxruntime`)</li><li>Asynchronous I/O with `aiohttp`, `anyio` for scalable network operations</li></ul> |
| 🛡️ | **Security**      | <ul><li>Uses `pydantic` for data validation and security</li><li>Secure handling of API keys and tokens (implied via environment/config)</li><li>Employs standard security practices for dependencies and network calls</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Heavy reliance on `requirements.txt` for dependency management</li><li>Includes ML, audio, NLP, and web libraries (`transformers`, `soundfile`, `requests`, `fastapi` implied)</li><li>Supports hardware-specific dependencies (`setuptools-rust`, `pywin32`) for cross-platform compatibility</li></ul> |

---

## Project Structure

```sh
└── luna-ai/
    ├── Hey-Luna.ppn
    ├── Luna.py
    ├── README.md
    ├── llm_interface.py
    ├── requirements.txt
    └── transcription.py
```
---

## Roadmap

- [X] **`Task 1`**: <strike>Advanced Memory Management.</strike>
- [ ] **`Task 2`**: Add Nari-Labs Dia 2.3 Model for Voice
- [ ] **`Task 3`**: Modular Features (Web Search, Code Generation, Image Interpretation)
- [ ] **`Task 4`**: Add Multi-Agent Capabilities

---
