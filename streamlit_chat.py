import random
import json
import torch
import streamlit as st
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

# Extract data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set bot name
bot_name = "Investo"

# Configure Streamlit page
st.set_page_config(
    page_title="Chat with Investo",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Streamlit UI
st.title("Chat with Investo")

user_input = st.text_input("You: ", "")
output_area = st.empty()

if st.button("Send"):
    with st.spinner("Thinking..."):
        if user_input.lower() == "quit":
            st.success(f"{bot_name}: Goodbye!")
        else:
            sentence = tokenize(user_input)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        st.text(f"{bot_name}: {random.choice(intent['responses'])}")
            else:
                st.text(f"{bot_name}: I do not understand...")