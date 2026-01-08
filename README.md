# 🎙️Voice Assistant AI

A sophisticated, voice-driven AI assistant built with Python that leverages **ElevenLabs Conversational AI** for natural voice interactions. This assistant doesn't just talk—it takes action, allowing you to generate images, create websites, and search the web using only your voice.

---

## ✨ Key Features

* [cite_start]**Natural Voice Interaction**: Real-time, low-latency voice conversations powered by ElevenLabs[cite: 1, 2].
* **AI Image Generation**: Integrated with **Stability AI** to create high-quality images (SDXL 1.0) at a fraction of the cost of other providers.
* **Web Development**: Automatically generates and saves valid HTML files based on your verbal descriptions.
* **Web Search**: Real-time information retrieval using DuckDuckGo to answer current questions.
* **File Management**: Ability to save notes and data directly to `.txt` files.

---

## 🛠️ Architecture & Tools


### Core Technologies
| Component | Technology Used |
| :--- | :--- |
| **Voice Engine** | [cite_start]ElevenLabs Conversational AI [cite: 1, 2] |
| **Image Generation** | Stability AI (SDXL 1.0) |
| **Search Engine** | DuckDuckGo Search API |
| **Orchestration** | [cite_start]Python & LangChain Community  |
| **Audio Interface** | [cite_start]PyAudio & ElevenLabs SDK  |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed and a working microphone.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirement.txt
