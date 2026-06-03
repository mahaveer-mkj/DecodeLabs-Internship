# 🤖 Project 1: Rule-Based AI Chatbot (WavYy)

### DecodeLabs AI Internship - Week 1

Welcome to **WavYy**, a simple Rule-Based AI Chatbot built as part of my first project during the DecodeLabs AI Internship.

This project focuses on understanding the foundations of Artificial Intelligence through **control flow, decision-making, and rule-based logic**. Instead of using Machine Learning or Large Language Models, WavYy responds to users based on predefined rules and a structured knowledge base.

---

## 🎯 Project Objective

The goal of this project was to learn how a machine can simulate conversation using:

* Conditional statements
* Control flow
* Infinite loops
* Dictionaries (Hash Maps)
* User input handling
* Decision-making logic

This project helped me understand the core principles behind how intelligent systems process information before moving on to advanced AI concepts.

---

## 🚀 Features

✅ Handles greetings and basic conversations

✅ Responds to predefined questions

✅ Uses a dictionary-based knowledge base

✅ Performs input sanitization using `.strip()` and `.lower()`

✅ Runs continuously until the user exits

✅ Provides a fallback response for unknown queries

---

## 🧠 Knowledge Areas Covered

WavYy can currently answer questions related to:

* Cricket
* Artificial Intelligence
* Python Programming
* GitHub
* General Introduction Questions

Example Questions:

```text
hello
how are you
what is your name
tell me about you
who is the king of cricket
who is the god of cricket
who won the ipl 2026
what is ai
what is python
what is github
who is your creator
bye
```

---

## 🏗️ Project Structure

```text
Project-1-RuleBasedChatbot/
│
├── chatbot.py
└── README.md
```

---

## ⚙️ Technologies Used

* Python 3
* Git
* GitHub

---

## 🔄 Working Principle

The chatbot follows the IPO (Input → Process → Output) model.

### Input

The user enters a message.

```python
user_message = input("You : ")
```

### Process

The input is cleaned and normalized.

```python
user_message = user_message.strip().lower()
```

The chatbot then checks whether the message exists in its knowledge base.

### Output

If a matching response is found, it is displayed to the user.

Otherwise, the chatbot returns a default fallback response.

---

## 💡 Key Learning Outcomes

Through this project, I learned:

* How rule-based AI systems work
* The importance of control flow in AI applications
* How dictionaries provide efficient data lookup
* How chatbots process and respond to user inputs
* The role of deterministic systems in Artificial Intelligence

---

## 📌 Future Improvements

Some features I plan to add in future versions:

* More conversational responses
* Multiple categories of knowledge
* Pattern matching
* Natural Language Processing (NLP)
* Integration with Machine Learning models

---

## 👨‍💻 Author

**Mahaveer Mundaluhari**

AI Intern @ DecodeLabs

Computer Science Engineering (AI & ML)

OUTR Bhubaneswar

---

### Week 1 Complete ✅

*"Every intelligent system starts with logic. Before teaching machines to learn, we must first teach them to think through rules."*
