# Project 1 : Rule-Based AI Chatbot

print("====================================")
print("      Welcome to WavYy     ")
print("====================================")
print("Type 'exit' anytime to close the chatbot....")
print()

# Knowledge Base using Dictionary
chat_responses = {
    "hello": "Hello! Nice to meet you.",
    "hi": "Hi! How are you today?",
    "good morning": "Good Morning! Have a great day.",
    "good afternoon": "Good Afternoon!",
    "good evening": "Good Evening!",
    "how are you": "I am fine. Thank you for asking.",
    "tell me about you": "I am WavYy, a Rule-Based AI Chatbot created for project work.",
    "what is your name": "My name is WavYy.",
    "what can you do": "I can answer simple questions according to my knowledge base and have suitable a conversation with you.",
    "what are the questions": "Well my knowledge base too small for now. But I'm preety sure that I can answer two famous questions: ' which chocolate is out stock now ' & ' which leak can't be filled up in India ' ",
    "which chocolate is out stock now": "That's World famous \"Melody\".",
    "which leak can't be filled up in India": "As per my knowledge, the 'NEET Paper leak' currently tops the list.",
    "who created you": "I was created as a Rule-Based AI Chatbot by Mahaveer.",
    "bye": "Goodbye...! Have a nice day."
}

# Infinite loop to keep chatbot running
while True:

    # Taking input from user
    user_message = input("You : ")

    # Input Sanitization and Normalization
    user_message = user_message.strip().lower()

    # Exit condition
    if user_message == "exit":
        print("ChatBot : Chat session ended.")
        break

    # Check if input exists in dictionary
    if user_message in chat_responses:

        response = chat_responses[user_message]

        print("ChatBot :", response)

    else:

        print("ChatBot : Sorry boss... that's beyond my knowledge base.")

print()
print("Bye...! Have a nice day...!")
