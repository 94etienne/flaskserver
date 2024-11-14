import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Initialize NLTK and Lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load chatbot training data from JSON
with open("chat.json", "r") as f:
    data = json.load(f)

# Preprocessing: Tokenization and Lemmatization
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in tokens])

X, y, classes = [], [], []

# Prepare training data
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(preprocess_sentence(pattern))
        y.append(intent['tag'])
    classes.append(intent['tag'])

# Vectorize input patterns
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Encode labels and train Logistic Regression
y_encoded = np.array([classes.index(tag) for tag in y])
model = LogisticRegression(max_iter=200)
model.fit(X_vectorized, y_encoded)

print("Chatbot is trained and ready!")

# Function to generate varied responses
def get_intent_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat function with improved logic
def chat():
    print("Start chatting! (Type 'quit' to exit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break

        user_input_processed = preprocess_sentence(user_input)
        input_vector = vectorizer.transform([user_input_processed])
        prediction = model.predict(input_vector)[0]
        tag = classes[prediction]

        # Get the appropriate response
        response = get_intent_response(tag)
        print(f"Bot: {response}")

chat()
