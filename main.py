from flask import Flask, jsonify, request
from flask_cors import CORS
import random, json, nltk, string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load intents from JSON file
with open("chat.json", "r") as f:
    intents = json.load(f)

# Initialize NLTK tokenizer, lemmatizer, stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function: Tokenization, Lemmatization, Stop-word removal, and Special character removal
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

# Prepare training data
X, y, classes = [], [], []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(preprocess_sentence(pattern))
        y.append(intent['tag'])
    classes.append(intent['tag'])

# Check class balance before resampling
print("Class distribution before resampling:", Counter(y))

# Vectorize input patterns using TF-IDF with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Encode labels as integers
y_encoded = np.array([classes.index(tag) for tag in y])

# Handle class imbalance with SMOTE (k_neighbors=1 for small datasets)
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y_encoded)

# Check class distribution after resampling
print("Class distribution after resampling:", Counter(y_resampled))

# Split data into training and testing sets (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.1, random_state=42
)

# Use RandomForest with hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Update hyperparameter grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Use the best model from GridSearchCV
model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Predict on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate and print accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Ensure classes align with the test labels
unique_test_classes = np.unique(y_test)
print("Unique classes in y_test:", unique_test_classes)

# Adjust target names to match the unique labels in y_test
target_names = [classes[i] for i in unique_test_classes]

# Display classification report and confusion matrix
print("Classification Report:\n", 
      classification_report(y_test, y_test_pred, target_names=target_names, labels=unique_test_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Perform cross-validation to evaluate model stability
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean() * 100:.2f}%")

# Retrieve response based on predicted tag
def get_intent_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# Flask route to handle user messages and predict intents
@app.route('/intents', methods=['GET'])
def get_intents():
    user_message = request.args.get('message', None)
    
    if user_message:
        # Preprocess and vectorize the user input
        user_message_processed = preprocess_sentence(user_message)
        input_vector = vectorizer.transform([user_message_processed])
        
        # Predict intent
        prediction = model.predict(input_vector)[0]
        tag = classes[prediction]
        
        # Get response based on predicted tag
        response = get_intent_response(tag)
        return jsonify({"message": response})
    
    return jsonify({"message": "No message provided. Please provide a valid user input."})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=4000)
