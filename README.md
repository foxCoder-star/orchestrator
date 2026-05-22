# Jarvis AI Assistant

A modular AI assistant built from scratch in Python — combining NLP, computer vision, memory, and orchestration into a single cohesive system.

---

## Project Vision

Jarvis is a personal ambient AI assistant inspired by systems like Apple Intelligence and J.A.R.V.I.S. from Iron Man. The goal is a context-aware, voice-enabled, modular assistant that can understand natural language, perceive its environment, remember past interactions, and execute actions autonomously.

This is not a chatbot wrapper. Every component is built and understood from the ground up.

---

## Architecture

```
Voice/Text ──→ NLP ──────┐
                          ├──→ Command Center ──→ executes action
Camera/Gesture ──→ CV ───┘         │
                                   │
                              Memory (persistent context)
                                   │
                         Logs & Validator (debugging + improvements)
```

Each module communicates through a shared common format, keeping components independent and swappable.

---

## Project Evolution

**Version 1 — Prototype**
The project began as a simple command-based assistant with basic NLP intent parsing, timer handling, and contextual follow-up detection. It worked but became difficult to scale as the vision grew.

**Version 2 — Modular Rebuild (current)**
After learning more about NLP and software architecture, the system was redesigned from scratch into a fully modular architecture. Each component now lives in its own module with a clean interface.

This redesign was driven by:
- The need for better context handling across multiple turns
- Wanting real ML-based intent classification instead of heuristics
- Separating concerns so each module can evolve independently
- Preparing the system for agentic and multimodal capabilities

---

## Current Status

| Module | Status |
|---|---|
| NLP — Intent Classification | ✅ Complete |
| NLP — Entity Extraction | ✅ Complete |
| NLP — Context Manager | ✅ Complete |
| Command Center | 🔄 In Progress |
| Memory | 🔲 Planned |
| Computer Vision | 🔲 Planned |
| Logs & Validator | 🔲 Planned |

---

## NLP Module

The NLP layer is the primary input port for voice and text. It processes raw input into structured data the Command Center can act on.

**Intent Classification**
- Fine-tuned `distilbert-base-uncased` on 300 labeled examples across 20 intent categories
- Confidence thresholding for unknown intent detection and clarification requests
- Trained locally — no data leaves the machine

**Entity Extraction**
- spaCy `en_core_web_sm` for named entity recognition (people, locations, organizations, dates)
- Regex fallback for structured patterns (durations, times, percentages)
- Intent-aware slot filtering — only extracts entities relevant to the detected intent

**Context Manager**
- 10-turn sliding window conversation history
- Accumulated slot tracking across turns
- Follow-up detection and intent inheritance
- Vague reference resolution ("it", "that", "this")

**Output format:**
```python
{
    "intent"                : "create_reminder",
    "confidence"            : 0.904,
    "entities"              : {"person": "Sarah", "time": "7pm", "relative_time": "tomorrow"},
    "requires_clarification": False,
    "context_used"          : False,
    "source"                : "nlp"
}
```

---

## Tech Stack

- Python 3.x
- HuggingFace Transformers — distilBERT fine-tuning
- spaCy — named entity recognition
- PyTorch — model training and inference
- sentence-transformers — (prototype phase)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/jarvis-ai.git
cd jarvis-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# Train the intent classifier
cd nlp
python3 train_intent.py

# Run the NLP test interface
python3 test.py
```

---

## Roadmap

- [x] Prototype — command parsing and basic intent detection
- [x] NLP layer — ML intent classification, entity extraction, multi-turn context
- [ ] Command Center — orchestration hub connecting all modules
- [ ] Memory — persistent context across sessions
- [ ] Computer Vision — gesture and camera-based input
- [ ] Logs & Validator — system monitoring and debugging
- [ ] Voice integration — speech recognition and TTS output
- [ ] Ambient UI — Apple Intelligence-style visual state feedback

---

## Author

Built by David Mishael — a high school student teaching himself AI systems, NLP, and software architecture through iterative project building.
