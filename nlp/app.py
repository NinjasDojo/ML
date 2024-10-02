import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('bilstm_lstm_model.h5')

# Define token2idx and idx2tag mappings (These must match what was used during training)
token2idx = {'example': 1, 'tokens': 2, 'more': 3}  # Replace with actual token mappings
idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG'}  # Replace with actual tag mappings

input_length = 100  # Replace with the actual input length used during training

# Function to preprocess input text (convert text to token indices and pad it)
def preprocess_input(text):
    tokens = [token2idx.get(word, token2idx['example']) for word in text.split()]  # Convert words to indices
    pad_tokens = pad_sequences([tokens], maxlen=input_length, padding='post', value=len(token2idx)-1)  # Pad tokens
    return pad_tokens

# Function to get the named entity tags from the model's predictions
def get_ner_tags(predictions):
    predicted_tags = np.argmax(predictions, axis=-1)  # Get the index of the highest probability tag
    tags = [idx2tag.get(tag_idx, 'O') for tag_idx in predicted_tags[0]]  # Map indices back to tag labels
    return tags

# Function for interactive NER
def interactive_ner():
    print("Enter a sentence for Named Entity Recognition (type 'exit' to quit):")

    while True:
        # Get input from the user
        sentence = input("Input sentence: ")

        # Exit if the user types 'exit'
        if sentence.lower() == 'exit':
            print("Exiting...")
            break

        # Preprocess the input sentence
        input_tokens = preprocess_input(sentence)

        # Make prediction using the model
        predictions = model.predict(input_tokens)

        # Get the predicted tags
        predicted_tags = get_ner_tags(predictions)

        # Print the results
        print("Input Sentence:", sentence)
        print("Predicted NER Tags:", predicted_tags)
        print()

# Run the interactive NER loop
if __name__ == "__main__":
    interactive_ner()
