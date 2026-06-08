<div align="center">

# 🤖 WavYy — Rule-Based AI Chatbot
### DecodeLabs Training Kit — Project 1: Foundations of Intelligent Systems

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Rule-Based](https://img.shields.io/badge/AI-Rule--Based-805AD5?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)

**A production-grade, fully documented Rule-Based AI chatbot —  
mastering core decision-making logic before stepping into learning algorithms.**

[How to Run](#-quick-start) · [Conversation Example](#-example-conversation) · [Key Concepts](#-key-concepts-explained)

---

</div>

## 📌 What This Project Does

WavYy is a deterministic chatbot that responds to user input using **predefined rules** and a **structured knowledge base**. It is designed to demonstrate the foundational principles of artificial intelligence — control flow, conditional logic, and data lookup — without any machine learning or external APIs.

| Stage | Task | Key Decision |
|-------|------|-------------|
| **Input** | Accept user message via the terminal | Readable, interactive loop — no GUI needed |
| **Process** | Sanitize input → lookup in knowledge base dictionary | `.strip().lower()` ensures case‑insensitive, whitespace‑tolerant matching |
| **Output** | Return the matched response or a fallback string | Graceful handling of unknown queries — no crashes, no silence |

> **Design Philosophy:** Every line of code is explicit. The chatbot never guesses — it follows strict, human‑defined logic. This project lays the mental model for how machines process language before they ever “learn.”

---

## 💬 Example Conversation

```text
You : hello
WavYy : Hi there! How can I help you?

You : what is your name
WavYy : I'm WavYy, your rule-based AI assistant.

You : who is the god of cricket
WavYy : Sachin Tendulkar is widely considered the God of Cricket.

You : tell me a joke
WavYy : I'm still learning. Please ask me something else!

You : bye
WavYy : Goodbye! Have a great day.
```

> ⚠️ **Deterministic by design:** WavYy never hallucinates. If a query is not in its knowledge base, it always responds with a fallback message — this is a fundamental trait of rule‑based systems.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mahaveer-mkj/decodelabs-rulebased-chatbot.git
cd decodelabs-rulebased-chatbot
```

### 2. No external dependencies required
The chatbot uses only the Python standard library. Simply ensure Python 3.9+ is installed.

### 3. Run the chatbot
```bash
python chatbot.py
```

The script will start an interactive session. Type your messages and press Enter. Type `bye`, `exit`, or `quit` to end the conversation.

---

## 📁 Project Structure

```
decodelabs-rulebased-chatbot/
│
├── chatbot.py        # Main chatbot script (single file)
└── README.md         # This file
```

The entire logic — input loop, sanitization, knowledge base, and response engine — lives in one clean, thoroughly commented Python file.

---

## 🏗️ Architecture Deep Dive

```
┌──────────────────────────────────────────────────────────────┐
│                    MAIN PROGRAM LOOP                         │
│                                                              │
│   while True:                                                │
│       user_input = input("You : ")                           │
│       cleaned    = user_input.strip().lower()                │
│                                                              │
│       if cleaned in exit_commands:  →  break (bye)           │
│       else if cleaned in knowledge_base:  →  print(response) │
│       else:  →  print(fallback_response)                     │
└──────────────────────────────────────────────────────────────┘
```

**How it works:**
1. **Infinite loop** keeps the conversation alive until an exit condition is met.
2. **Input sanitization** strips accidental whitespace and normalises case — critical for dictionary lookups where `"Hello"` ≠ `"hello"`.
3. **Hash map (dictionary)** stores question‑response pairs for O(1) average lookup — the same data structure powering many production chatbots.
4. **Fallback mechanism** ensures the bot always replies, even when it doesn’t understand the query.

---

## ⚙️ Configurable Parameters

The knowledge base is a standard Python dictionary. Extending the chatbot is as simple as adding new key‑value pairs:

```python
knowledge_base = {
    "hello": "Hi there! How can I help you?",
    "what is your name": "I'm WavYy, your rule-based AI assistant.",
    "who won the ipl 2026": "Delhi Capitals won the IPL 2026 title.",
    # Add any number of custom Q&A pairs here
}
```

You can also customise the **exit commands** and **fallback response** at the top of `chatbot.py`.

---

## 🧠 Key Concepts Explained

### Why rule‑based?
Before teaching a machine to learn from data, we must first teach it to follow explicit logic. Rule‑based systems are:
- **Deterministic** — same input always produces the same output.
- **Transparent** — every response is traceable to a human‑written rule.
- **Controlled** — no risk of generating inappropriate or incorrect information.

### Why `.strip().lower()`?
Users may type `"   Hello   "` or `"HELLO"`. Without normalisation, the string would fail a dictionary lookup for `"hello"`. This tiny preprocessing step dramatically improves the bot’s perceived intelligence.

### Why a hash map (dictionary)?
Python dictionaries provide constant‑time average lookup. A list of `if-elif` chains would grow slower as the knowledge base expands; the dictionary scales efficiently.

### Why a fallback response?
Real users ask anything. An AI that goes silent when confused breaks the conversational illusion. A consistent fallback like *"I’m still learning. Please ask me something else!"* keeps the interaction alive while clearly signalling the system’s limits.

---

## 🛠️ Tech Stack

| Library     | Version | Purpose                         |
|-------------|---------|---------------------------------|
| `Python`    | 3.9+    | Core language — no external libs |
| `Git`       | —       | Version control                  |
| `GitHub`    | —       | Hosting and portfolio            |

---

## 📚 Project Context

This project is the first step in the **DecodeLabs Training Kit**, developed during my AI internship at [DecodeLabs](https://www.decodelabs.tech/). Project 1 establishes the critical mental model: **intelligent systems start with logic, not statistics.** It bridges the gap between beginner programming and the supervised learning pipeline built in Project 2.

---

## 👤 Author

**Mahaveer Mundaluhari**  
*Artificial Intelligence Intern @ [DecodeLabs](https://www.decodelabs.tech/)*  
*B.S. Data Science & Applications — IIT Madras*  
*B.Tech CSE (AI & ML) — OUTR Bhubaneswar*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mahaveer%20Mundaluhari-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mahaveer-mundaluhari/)
[![GitHub](https://img.shields.io/badge/GitHub-mahaveer--mkj-181717?style=flat&logo=github)](https://github.com/mahaveer-mkj)
[![Email](https://img.shields.io/badge/Email-mahaveer%40maxiwoxi.com-EA4335?style=flat&logo=gmail)](mailto:mahaveer@maxiwoxi.com)

---

<div align="center">

*Built with logic. Documented with clarity.*  
**Week 1 Complete ✅ — Every intelligent system starts with rules.**

</div>
