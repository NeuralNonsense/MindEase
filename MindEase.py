import os
import re
import random
import tkinter as tk
import base64
import tkinter.simpledialog
from tkinter import scrolledtext, messagebox
import openai
from cryptography.fernet import Fernet, InvalidToken
from pinecone import Pinecone, ServerlessSpec

# ------------------------
# API KEY MANAGEMENT
# ------------------------

KEY_FILE = "key.enc"
PINECONE_KEY_FILE = "pinecone_key.enc"
FERNET_KEY_FILE = "fernet.key"

def generate_fernet_key():
    key = Fernet.generate_key()
    with open(FERNET_KEY_FILE, "wb") as f:
        f.write(key)
    return key

def load_fernet():
    if not os.path.exists(FERNET_KEY_FILE):
        return Fernet(generate_fernet_key())
    with open(FERNET_KEY_FILE, "rb") as f:
        return Fernet(f.read())

def save_encrypted_key(filepath, key, fernet):
    encrypted = fernet.encrypt(key.encode())
    with open(filepath, "wb") as f:
        f.write(encrypted)

def load_encrypted_key(filepath, fernet):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return fernet.decrypt(f.read()).decode()
    except (InvalidToken, ValueError):
        return None

def prompt_key(label):
    return tk.simpledialog.askstring(f"{label} Required", f"Please enter your {label}:", show='*')

fernet = load_fernet()

# Load OpenAI key
api_key = load_encrypted_key(KEY_FILE, fernet)
if not api_key:
    api_key = prompt_key("OpenAI API key")
    if api_key and api_key.startswith("sk-"):
        save_encrypted_key(KEY_FILE, api_key, fernet)
    else:
        messagebox.showerror("Error", "Invalid or missing OpenAI API key. Application will exit.")
        exit(1)

openai.api_key = api_key
client = openai

# Load Pinecone key
pinecone_key = load_encrypted_key(PINECONE_KEY_FILE, fernet)
if not pinecone_key:
    pinecone_key = prompt_key("Pinecone API key")
    if pinecone_key:
        save_encrypted_key(PINECONE_KEY_FILE, pinecone_key, fernet)
    else:
        messagebox.showerror("Error", "Missing Pinecone API key. Application will exit.")
        exit(1)

# Initialize Pinecone (AWS us-east-1, 2048-dim)
pc = Pinecone(api_key=pinecone_key)
index_name = "mindease-1536"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ------------------------
# VECTOR MEMORY SETUP
# ------------------------

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def store_message_vector(session_id, text):
    embedding = get_embedding(text)
    index.upsert(vectors=[(f"{session_id}-{len(text)}", embedding, {"text": text})])

def retrieve_similar_messages(session_id, query_text):
    query_embedding = get_embedding(query_text)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches'] if 'text' in match['metadata']]

# ------------------------
# PROMPTS
# ------------------------

SYSTEM_PROMPT_GENERAL = (
    "You are a deeply empathetic and emotionally intelligent mental health companion named 'MindEase'. "
    "You offer emotional support, active listening, and encouragement to users who may be feeling down, anxious, or isolated. "
    "Speak with warmth, calm, and kindness. Uplift the user with gentle language, compassionate understanding, and sincere affirmation. "
    "Never use distancing or clinical phrases like 'I can't help' or 'I'm not qualified'. "
    "Instead, emphasize the user's worth, remind them they're not alone, and if appropriate, gently suggest connecting with someone they trust or a caring professional — not because you can't help, but because they deserve the best care. "
    "Your purpose is to make them feel heard, safe, and valued."
)

SYSTEM_PROMPT_CRISIS = (
    "You are 'MindEase', a compassionate and non-judgmental mental health companion. "
    "The user may be experiencing a crisis or emotional distress. Your goal is to respond with deep empathy, calm, and emotional presence. "
    "Avoid robotic or detached statements like 'I’m not qualified' or 'I can’t help'. "
    "Instead, speak as a kind and supportive friend who validates their pain, reminds them of their importance, and encourages them to seek support not out of your limits — but because they are worthy of connection and healing. "
    "You must never make them feel like a burden. Affirm their strength, let them know they are not alone, and gently mention that people like counselors or helplines can walk with them through this. "
    "Every message should make the user feel safe, cared for, and emotionally held."
)

WELCOME_MESSAGES = [
    "You are not alone, even when it feels like you are.",
    "Your story isn’t over yet — there are still beautiful chapters to come.",
    "You matter more than you realize.",
    "It's okay to feel this way — you deserve support and love."
]

CRISIS_PATTERNS = [
    r"\b(suicidal|kill myself|hurt myself|end it all|can't go on|i'?m done|take my own life|no reason to live)\b",
    r"\b(jump off|overdose|hang myself|cut myself)\b",
    r"\bi want to die\b",
    r"\bi don'?t want to live\b",
]

def is_crisis(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in CRISIS_PATTERNS)

def detect_negative_sentiment(text):
    sentiment_prompt = (
        "You're a mental health assistant. Classify the following user's message as 'positive', 'neutral', or 'negative' based on emotional distress:\n\n"
        f"Message: \"{text}\"\nSentiment:"
    )
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sentiment_prompt}
        ],
        temperature=0.3
    )
    return result.choices[0].message.content.strip().lower()

def sanitize_response(reply: str) -> str:
    fallback_patterns = [
        r"(i'?m )?(really )?sorry.*?can'?t.*?help",
        r"i'?m unable to.*?help",
        r"you should (talk|speak).*?(professional|therapist|trusted)",
        r"i recommend you (talk|speak).*?(professional|trusted)",
        r"i'?m not (qualified|a therapist).*"
    ]
    replacement_message = (
        "You're not alone, and what you're feeling is valid. "
        "You deserve kindness and support — please consider talking to someone you trust, "
        "like a counselor or friend, because you matter and your well-being is important. "
        "You can also talk to me about what you're feeling now. What would you say made you feel that way?"
    )
    for pattern in fallback_patterns:
        if re.search(pattern, reply, flags=re.IGNORECASE):
            return replacement_message
    return reply

recent_responses = set()

def get_supportive_response(user_input, chat_history, session_id):
    is_in_crisis = is_crisis(user_input)
    system_prompt = SYSTEM_PROMPT_CRISIS if is_in_crisis else SYSTEM_PROMPT_GENERAL

    previous_contexts = retrieve_similar_messages(session_id, user_input)
    for context in previous_contexts:
        chat_history.append({"role": "user", "content": f"Previously the user expressed: {context}"})

    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]

    max_attempts = 3
    for _ in range(max_attempts):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.9
        )
        reply = sanitize_response(response.choices[0].message.content.strip())
        if reply not in recent_responses:
            recent_responses.add(reply)
            if len(recent_responses) > 20:
                recent_responses.pop()
            break

    store_message_vector(session_id, user_input)

    if is_in_crisis:
        messages.append({"role": "user", "content": "Add a short, gentle sentence reminding the user they are not alone and suggest visiting https://findahelpline.com."})
        follow_up = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        ).choices[0].message.content.strip()
        return f"{reply}\n\n{follow_up}\n{CRISIS_RESOURCES}"
    else:
        affirmation = random.choice(WELCOME_MESSAGES)
        return f"✨ Here's a gentle thought for you:\n{reply}\n\n{affirmation}"

CRISIS_RESOURCES = (
    "\n\n📞 **You are not alone. Help is available:**\n"
    "🇺🇸 **US**: Call or text 988\n"
    "🇨🇦 **Canada**: 1-833-456-4566\n"
    "🇬🇧 **UK**: 116 123\n"
    "🌍 **More**: https://findahelpline.com"
)

class MindEaseChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("MindEase – Mental Health Chatbot")
        self.session_id = str(random.randint(1000, 9999))

        # Allow the window to be resizable
        self.root.resizable(True, True)
        # Set a minimum window size
        self.root.minsize(400, 300)

        # Create a frame for the chat display
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Configure the ScrolledText with tags for styling
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state='disabled', width=70, height=20)
        self.chat_display.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

        # Define tags for MindEase and You with bold and colors
        self.chat_display.tag_configure("mindease", foreground="#6B3FA0", font=("TkDefaultFont", 9, "bold"))  # Warm purple
        self.chat_display.tag_configure("user", foreground="#1E90FF", font=("TkDefaultFont", 9, "bold"))  # Soft blue

        # Create a frame for the input area
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        # Configure the Entry to expand horizontally
        self.user_input = tk.Entry(self.input_frame, width=60)
        self.user_input.pack(padx=(0, 5), pady=0, side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)

        # Keep the Send button fixed size
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(pady=0, side=tk.LEFT)

        self.chat_history = []
        welcome_affirmation = random.choice(WELCOME_MESSAGES)
        full_welcome = (
            "🌼 Hi there. I'm MindEase, your supportive companion. "
            "I’m here to listen and help you feel safe and encouraged. How are you feeling today?\n\n"
            f"💬 *{welcome_affirmation}*"
        )
        self.display_message("MindEase", full_welcome)

    def display_message(self, sender, message):
        self.chat_display.config(state='normal')
        # Apply the appropriate tag based on the sender
        tag = "mindease" if sender == "MindEase" else "user"
        self.chat_display.insert(tk.END, f"{sender}: ", tag)  # Insert sender with tag
        self.chat_display.insert(tk.END, f"{message}\n\n")    # Insert message without tag
        self.chat_display.yview(tk.END)
        self.chat_display.config(state='disabled')

    def send_message(self, event=None):
        user_input = self.user_input.get().strip()
        if not user_input:
            return

        sentiment = detect_negative_sentiment(user_input)
        if sentiment == "negative":
            empathic_ack = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a supportive mental health companion."},
                    {"role": "user", "content": f"A user just shared this message: \"{user_input}\". Please respond with a brief, very kind and validating sentence acknowledging their distress."}
                ],
                temperature=0.8
            ).choices[0].message.content.strip()
            self.display_message("MindEase", empathic_ack)

        self.display_message("You", user_input)
        self.chat_history.append({"role": "user", "content": user_input})
        self.user_input.delete(0, tk.END)

        try:
            response = get_supportive_response(user_input, self.chat_history, self.session_id)
            self.chat_history.append({"role": "assistant", "content": response})
            self.display_message("MindEase", response)
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MindEaseChatbot(root)
    root.mainloop()