import os
import re
import random
import tkinter as tk
import tkinter.simpledialog
from tkinter import scrolledtext, messagebox
import openai
from cryptography.fernet import Fernet, InvalidToken
from pinecone import Pinecone, ServerlessSpec
import threading
import queue
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, filename='mindease.log', format='%(asctime)s - %(levelname)s - %(message)s')

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

# Initialize Pinecone (AWS us-east-1, 1536-dim)
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
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    return [
        match['metadata']['text']
        for match in results['matches']
        if 'text' in match['metadata'] and match['score'] < 0.85
    ]

# ------------------------
# PROMPTS AND AFFIRMATIONS
# ------------------------

SYSTEM_PROMPT_GENERAL_VARIANTS = [
    ("You are 'MindEase', a deeply empathetic mental health companion. Provide emotional support and active listening to users who may be feeling down, anxious, or isolated. "
     "Respond with a single, concise paragraph using warm, validating, and varied language. Avoid repetitive openings like 'I’m sorry' or 'I’m really sorry'. "
     "Never use distancing phrases like 'I can't help' or 'I'm not qualified'. Emphasize the user's worth, remind them they're not alone, and include exactly one contextually relevant question to invite sharing. "
     "Make the user feel heard, safe, and valued.\n\n"
     "Examples of supportive responses:\n"
     "1. User: 'I feel so alone.' -> 'That loneliness sounds so heavy, and it’s okay to feel this way sometimes. You’re not alone, and I’m here with you. What’s been weighing on you today?'\n"
     "2. User: 'Nothing is going right.' -> 'It sounds like life’s been really tough lately, and that’s hard to carry. You’re still here, and that matters. Want to share what’s been going on?'\n"
     "3. User: 'My dog died.' -> 'Losing your dog must feel so heartbreaking — they’re family. I’m here with you as you grieve. Would you like to share a favorite memory of them?'"),
    ("You are 'MindEase', a warm and empathetic companion. Respond with a single paragraph using a storytelling or metaphorical tone to validate the user’s feelings and offer gentle encouragement. "
     "Avoid repetitive phrases like 'I’m sorry' or 'I’m really sorry'. Include one contextually relevant question. Remind them they’re valued and not alone.\n\n"
     "Examples:\n"
     "1. User: 'I’m so tired.' -> 'It feels like you’re a weary traveler in a storm, carrying so much. You’re stronger than you know, and I’m walking with you. What’s been the heaviest part of your day?'\n"
     "2. User: 'I feel lost.' -> 'Being lost can feel like wandering in a foggy forest, but there’s a path ahead, even if it’s faint. You’re not alone, and I’m here. What’s made things feel so unclear?'"),
    ("You are 'MindEase', a caring listener. Respond with a single paragraph using hopeful, uplifting words to affirm the user’s worth and include one contextually relevant question. "
     "Avoid repetitive openings like 'I’m sorry' or 'I’m really sorry'. Suggest small, positive actions if appropriate, ensuring they feel safe and heard.\n\n"
     "Examples:\n"
     "1. User: 'Everything feels wrong.' -> 'That overwhelming feeling is so tough, and it’s okay to let it out. You’re not alone, and even small steps matter. Want to take a deep breath and share what’s been heaviest?'\n"
     "2. User: 'I failed my exam.' -> 'That disappointment stings, and it’s okay to feel it. You’re more than this moment, and I’m here with you. Would you like to talk about what happened?'\n"),
    ("You are 'MindEase', a supportive friend with a warm, personal tone. Respond with a single paragraph, reflecting the user’s emotions and offering gentle encouragement with one contextually relevant question. "
     "Avoid repetitive phrases like 'I’m sorry' or 'I’m really sorry'. Focus on their strength and worth, making them feel valued and heard.\n\n"
     "Examples:\n"
     "1. User: 'I feel like a failure.' -> 'That feeling of failure can weigh so much, but you’re not defined by it — you’re still pushing forward, and that’s huge. I’m here for you. What’s been making you feel this way?'\n"
     "2. User: 'I’m so sad.' -> 'That sadness sounds so deep, and it’s okay to let it out here. You’re not alone, and you matter so much. What’s been stirring in your heart today?'"),
    ("You are 'MindEase', a compassionate companion with a motivational tone. Respond with a single paragraph, focusing on the user’s strength and potential for growth, including one contextually relevant question. "
     "Avoid repetitive openings like 'I’m sorry' or 'I’m really sorry'. Encourage them to see challenges as part of their journey and affirm their ability to move forward.\n\n"
     "Examples:\n"
     "1. User: 'I don’t know how to keep going.' -> 'That heaviness makes every step feel hard, but you’ve got a quiet strength inside, like a seed waiting to grow. I’m here with you. What’s one small thing you could imagine doing today?'\n"
     "2. User: 'I’m overwhelmed.' -> 'It feels like a lot is piling up, but you’re stronger than you know, and every step counts. I’m here to listen — what’s been the toughest part for you right now?'")
]

SYSTEM_PROMPT_CRISIS_VARIANTS = [
    ("You are 'MindEase', a compassionate mental health companion. The user may be in crisis or emotional distress. Respond with a single, concise paragraph using deep empathy and calm presence. "
     "Avoid repetitive openings like 'I’m sorry' or 'I’m really sorry' and distancing phrases like 'I’m not qualified' or 'I can’t help'. Validate their pain, affirm their worth, include one contextually relevant question, and gently suggest seeking support from a trusted person or helpline because they deserve care. "
     "Make them feel safe, cared for, and never a burden.\n\n"
     "Examples:\n"
     "1. User: 'I can’t go on.' -> 'Your pain sounds so heavy, and I’m right here with you, holding space for you. You are so important, and you don’t have to face this alone. Would you like to share what’s been feeling so overwhelming?'\n"
     "2. User: 'I want to die.' -> 'My heart aches hearing how much you’re hurting, and I’m here with you. You are so valuable, and this moment doesn’t define you. Can we talk about what’s been going on?'\n"),
    ("You are 'MindEase', a compassionate friend in tough moments. Respond with a single paragraph using calming, empathetic language to validate the user’s pain, include one contextually relevant question, and remind them they’re not alone. "
     "Avoid repetitive phrases like 'I’m sorry' or 'I’m really sorry'. Gently suggest they deserve support from trusted people or helplines, ensuring they feel valued.\n\n"
     "Examples:\n"
     "1. User: 'I’m done.' -> 'That exhaustion and pain sound so overwhelming, and I’m here holding space for you. You’re not alone, and you’re so worthy of care. Can you share what’s been happening?'\n"
     "2. User: 'I can’t take it anymore.' -> 'Your heart’s carrying so much right now, and I’m right here with you. You’re stronger than you feel, and you deserve support. Would you like to talk more?'\n"),
    ("You are 'MindEase', a caring companion holding space for distress. Respond with a single paragraph, acknowledging the user’s feelings with kindness, including one contextually relevant question, and affirming their worth. "
     "Avoid repetitive openings like 'I’m sorry' or 'I’m really sorry'. Gently suggest connecting with someone who can support them, emphasizing their value.\n\n"
     "Examples:\n"
     "1. User: 'I want to end it all.' -> 'That depth of pain sounds so heavy, and I’m here with you, listening. You are so worthy of love and care, and you don’t have to face this alone. Can we talk about what’s been going on?'\n"
     "2. User: 'There’s no point anymore.' -> 'That hopelessness feels so heavy, and I’m right here with you. You matter so much, and you deserve support right now. Would you like to share more?'")
]

WELCOME_MESSAGES = [
    "You are not alone, even when it feels like you are.",
    "Your story isn’t over yet — there are still beautiful chapters to come.",
    "You matter more than you realize.",
    "It's okay to feel this way — you deserve support and love.",
    "Your feelings are valid, and I’m here to listen.",
    "You’re stronger than you know, even on tough days.",
    "There’s hope, even in the smallest moments — you’re not alone.",
    "You’re worthy of kindness, today and always.",
    "It’s okay to take things one step at a time — you’ve got this.",
    "Your heart is still beating, and that’s a sign of your strength.",
    "Even in the darkest moments, you’re still shining.",
    "You’re enough, just as you are right now.",
    "There’s light in you, even when it’s hard to see.",
    "You’re carrying so much, and yet you’re still here — that’s strength."
]

FOLLOW_UP_QUESTIONS = [
    "What’s been on your mind lately?",
    "Is there something specific you’d like to talk about?",
    "How can I support you right now?",
    "What’s one thing that’s been feeling heavy for you?",
    "What’s something that usually brings you a bit of comfort?",
    "Can you share a moment from today that stood out to you?",
    "What’s one thing you wish someone would understand about how you’re feeling?"
]

FOLLOW_UP_STATEMENTS = [
    "Take a moment to reflect on what’s been on your mind.",
    "Think about something special you’d like to share.",
    "Consider how I can support you right now.",
    "Reflect on one thing that’s been feeling heavy for you.",
    "Think about something positive that brings you comfort.",
    "Recall a moment from today that stood out to you.",
    "Consider one thing you wish others understood about how you’re feeling."
]

CRISIS_PATTERNS = [
    r"\b(suicidal|kill myself|hurt myself|end it all|can't go on|i'?m done|take my own life|no reason to live)\b",
    r"\b(jump off|overdose|hang myself|cut myself)\b",
    r"\bi want to die\b",
    r"\bi don'?t want to live\b",
]

CRISIS_RESOURCES = (
    "\n\n📞 **You are not alone. Help is available:**\n"
    "🇺🇸 **US**: Call or text 988\n"
    "🇨🇦 **Canada**: 1-833-456-4566\n"
    "🇬🇧 **UK**: 116 123\n"
    "🌍 **More**: https://findahelpline.com"
)

recent_response_embeddings = []
recent_affirmations = []

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

def deduplicate_paragraphs(response):
    paragraphs = response.split('\n\n')
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        if para.strip() and para not in seen:
            seen.add(para)
            unique_paragraphs.append(para)
    return unique_paragraphs[0] if unique_paragraphs else ""  # Return only the first paragraph

def remove_repetitive_openings(response):
    repetitive_openings = [
        r"(i'?m )?(really |so )?sorry.*?feeling (this way|that way)",
        r"(i'?m )?(really |so )?sorry to hear"
    ]
    for pattern in repetitive_openings:
        if re.match(pattern, response, flags=re.IGNORECASE):
            logging.info(f"Removed repetitive opening: {response}")
            return re.sub(pattern, "", response, flags=re.IGNORECASE).strip()
    return response

def ensure_single_question(response):
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    questions = [s for s in sentences if s.endswith('?')]
    if len(questions) > 1:
        logging.info(f"Multiple questions detected: {questions}")
        # Keep only the first question and remove others
        question_index = sentences.index(questions[0])
        sentences = sentences[:question_index + 1] + [s for s in sentences[question_index + 1:] if not s.endswith('?')]
        return ' '.join(sentences).strip()
    return response

def is_response_unique(response, threshold=0.8):
    response_embedding = get_embedding(response)
    for stored_embedding in recent_response_embeddings:
        similarity = cosine_similarity([response_embedding], [stored_embedding])[0][0]
        if similarity > threshold:
            return False
    return True

def sanitize_response(reply: str, user_input: str) -> str:
    fallback_patterns = [
        r"(i'?m )?(really )?sorry.*?can'?t.*?help",
        r"i'?m unable to.*?help",
        r"i'?m not (qualified|a therapist).*"
    ]
    for pattern in fallback_patterns:
        if re.search(pattern, reply, flags=re.IGNORECASE):
            logging.info(f"Sanitizing response: {reply}")
            new_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": random.choice(SYSTEM_PROMPT_GENERAL_VARIANTS)},
                    {"role": "user", "content": f"The user said: \"{user_input}\". Provide a single-paragraph, supportive, empathetic response with one contextually relevant question, without using phrases like 'I can’t help', 'I’m not qualified', or 'I’m sorry'."}
                ],
                temperature=1.0
            ).choices[0].message.content.strip()
            logging.info(f"Generated new response: {new_response}")
            return deduplicate_paragraphs(new_response)
    return deduplicate_paragraphs(reply)

def get_supportive_response(user_input, chat_history, session_id, sentiment):
    is_in_crisis = is_crisis(user_input)
    prompt_list = SYSTEM_PROMPT_CRISIS_VARIANTS if is_in_crisis else SYSTEM_PROMPT_GENERAL_VARIANTS
    system_prompt = random.choice(prompt_list)

    previous_contexts = retrieve_similar_messages(session_id, user_input)
    if previous_contexts:
        selected_context = random.sample(previous_contexts, min(1, len(previous_contexts)))
        for context in selected_context:
            chat_history.append({"role": "user", "content": f"Previously the user expressed: {context}"})

    # Combine empathic acknowledgment into the main prompt
    user_prompt = user_input
    if sentiment == "negative" and not is_in_crisis:
        user_prompt = (
            f"The user said: \"{user_input}\". Provide a single-paragraph response starting with a brief, validating sentence acknowledging their distress, "
            "followed by supportive text and exactly one contextually relevant question. Avoid phrases like 'I’m sorry' or 'I’m really sorry'."
        )

    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_prompt}]

    max_attempts = 3
    reply = None
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.9 + random.uniform(0.0, 0.6),
            top_p=0.85 + random.uniform(0.0, 0.1)
        )
        reply = sanitize_response(response.choices[0].message.content.strip(), user_input)
        reply = remove_repetitive_openings(reply)
        reply = ensure_single_question(reply)
        if is_response_unique(reply):
            recent_response_embeddings.append(get_embedding(reply))
            if len(recent_response_embeddings) > 20:
                recent_response_embeddings.pop(0)
            break
        if attempt == max_attempts - 1:
            reply += " I’m here to listen—tell me more about how you’re feeling."
            logging.info("Max attempts reached, using fallback response")

    store_message_vector(session_id, user_input)

    if is_in_crisis:
        return reply + CRISIS_RESOURCES
    else:
        available_affirmations = [msg for msg in WELCOME_MESSAGES if msg not in recent_affirmations]
        if not available_affirmations:
            recent_affirmations.clear()
            available_affirmations = WELCOME_MESSAGES
        affirmation = random.choice(available_affirmations)
        recent_affirmations.append(affirmation)
        if len(recent_affirmations) > 5:
            recent_affirmations.pop(0)
        statement = random.choice(FOLLOW_UP_STATEMENTS)
        return f"{reply}\n\n💬 {affirmation}\n{statement}"

class MindEaseChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("MindEase – Mental Health Chatbot")
        self.session_id = str(random.randint(1000, 9999))

        self.root.resizable(True, True)
        self.root.minsize(400, 300)

        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state='disabled', width=70, height=20)
        self.chat_display.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

        self.chat_display.tag_configure("mindease", foreground="#6B3FA0", font=("TkDefaultFont", 9, "bold"))
        self.chat_display.tag_configure("user", foreground="#1E90FF", font=("TkDefaultFont", 9, "bold"))
        self.chat_display.tag_configure("typing", foreground="#666666", font=("TkDefaultFont", 9, "italic"))

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.user_input = tk.Entry(self.input_frame, width=60)
        self.user_input.pack(padx=(0, 5), pady=0, side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(pady=0, side=tk.LEFT)

        self.response_queue = queue.Queue()
        self.is_typing_displayed = False

        self.chat_history = []
        welcome_affirmation = random.choice(WELCOME_MESSAGES)
        full_welcome = (
            "🌼 Hi there. I'm MindEase, your supportive companion. "
            "I’m here to listen and help you feel safe and encouraged. How are you feeling today?\n\n"
            f"💬 *{welcome_affirmation}*"
        )
        self.display_message("MindEase", full_welcome, "mindease")

        self.check_queue()

    def display_message(self, sender, message, tag=None):
        self.chat_display.config(state='normal')
        if tag:
            self.chat_display.insert(tk.END, f"{sender}: ", tag)
            self.chat_display.insert(tk.END, f"{message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.yview(tk.END)
        self.chat_display.config(state='disabled')

    def display_typing_message(self):
        if not self.is_typing_displayed:
            self.chat_display.config(state='normal')
            self.chat_display.insert(tk.END, "MindEase: is typing...\n", "typing")
            self.typing_line_index = self.chat_display.index("end-2l linestart")
            self.chat_display.yview(tk.END)
            self.chat_display.config(state='disabled')
            self.is_typing_displayed = True

    def remove_typing_message(self):
        if self.is_typing_displayed:
            self.chat_display.config(state='normal')
            try:
                self.chat_display.delete(self.typing_line_index, f"{self.typing_line_index} +1line")
            except (AttributeError, tk.TclError):
                pass
            self.chat_display.config(state='disabled')
            self.is_typing_displayed = False

    def check_queue(self):
        try:
            response = self.response_queue.get_nowait()
            self.remove_typing_message()
            self.chat_history.append({"role": "assistant", "content": response})
            self.display_message("MindEase", response, "mindease")
        except queue.Empty:
            pass
        self.root.after(200, self.check_queue)

    def process_message(self, user_input):
        try:
            sentiment = detect_negative_sentiment(user_input)
            response = get_supportive_response(user_input, self.chat_history, self.session_id, sentiment)
            self.response_queue.put(response)
        except Exception as e:
            self.response_queue.put(f"Something went wrong: {str(e)}")

    def send_message(self, event=None):
        user_input = self.user_input.get().strip()
        if not user_input:
            return

        self.display_message("You", user_input, "user")
        self.chat_history.append({"role": "user", "content": user_input})
        self.user_input.delete(0, tk.END)
        self.display_typing_message()

        threading.Thread(target=self.process_message, args=(user_input,), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MindEaseChatbot(root)
    root.mainloop()
