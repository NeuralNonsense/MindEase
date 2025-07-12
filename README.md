MindEase - Mental Health Chatbot

MindEase is a compassionate, desktop-based mental health chatbot built with Python and Tkinter. It provides emotional support and encouragement through empathetic conversations, leveraging OpenAI's language models for natural language understanding and Pinecone for vector-based memory to recall relevant past interactions. MindEase aims to create a safe, non-judgmental space for users to express their feelings, with special handling for crisis situations.

Features





Empathetic Conversations: Responds with warmth and kindness, offering affirmations and support.



Crisis Detection: Identifies distressing language and provides tailored responses with crisis resources (e.g., helplines).



Vector Memory: Stores and retrieves relevant past messages using Pinecone for context-aware responses.



Secure API Key Storage: Encrypts OpenAI and Pinecone API keys using the cryptography library.



Responsive GUI: Built with Tkinter, featuring a resizable window, colored and bolded sender labels (MindEase in purple, user in blue), and a clean chat interface.



Sentiment Analysis: Detects negative sentiment to provide immediate empathetic acknowledgment.

Prerequisites





Python 3.12 or higher



OpenAI API key (get one from OpenAI)



Pinecone API key (get one from Pinecone)



Git (optional, for cloning the repository)

Installation





Clone the Repository (or download the source code):

git clone https://github.com/your-username/mindease.git
cd mindease



Install Dependencies: Ensure you have Python 3.12 installed, then install the required packages:

pip install -r requirements.txt

The requirements.txt includes:

openai>=1.30.0
cryptography>=42.0.0
pinecone>=5.0.0

Note: Tkinter is included in Python’s standard library and does not need to be installed.



Prepare API Keys:





You’ll be prompted to enter your OpenAI and Pinecone API keys when you first run the application.



These keys are encrypted and stored locally for future use.

Usage





Run the Application:

python MindEase.py



Enter API Keys:





On first run, input your OpenAI API key (starts with sk-) and Pinecone API key when prompted.



Invalid or missing keys will cause the application to exit with an error message.



Interact with MindEase:





The GUI opens with a welcome message from MindEase.



Type your message in the input field and press "Send" or hit Enter.



MindEase responds with supportive messages, styled with a purple "MindEase:" label.



Your messages are prefixed with a blue "You:" label.



The chat window is resizable, and the interface adapts dynamically.



Crisis Support:





If the chatbot detects crisis-related language (e.g., "I want to die"), it responds with extra care and provides helpline resources (e.g., 988 for the US, https://findahelpline.com for global options).

Screenshots

(Add screenshots of the GUI here, e.g., the chat window with colored labels. You can upload images to GitHub and link them here, like:)



Project Structure

mindease/
├── MindEase.py          # Main application script
├── requirements.txt     # Python dependencies
├── key.enc             # Encrypted OpenAI API key (generated on first run)
├── pinecone_key.enc    # Encrypted Pinecone API key (generated on first run)
├── fernet.key          # Encryption key for API keys (generated on first run)
├── README.md           # This file

Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a new branch (git checkout -b feature/your-feature).



Make your changes and commit (git commit -m "Add your feature").



Push to the branch (git push origin feature/your-feature).



Open a pull request.

Please ensure your code follows the project’s style and includes tests where applicable.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments





Built with OpenAI for natural language processing.



Uses Pinecone for vector storage and retrieval.



Powered by Python and Tkinter.



Inspired by the need for accessible, empathetic mental health support.

Disclaimer

MindEase is not a replacement for professional mental health care. It is designed to provide emotional support and encouragement. In case of a mental health crisis, please contact a professional or use the provided helpline resources.



If you encounter issues or have questions, open an issue on this repository or contact the maintainer.