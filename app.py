import os
import pickle
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load the trained model
model = load_model('model.h5')

# Load the word index
with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# Reverse the word index to create an index to word mapping
index_word = {index: word for word, index in word_index.items()}

# Load the feature extractor (assuming InceptionV3 was used during training)
feature_extractor = InceptionV3(include_top=False, pooling='avg')

base_model = InceptionV3(weights='imagenet', include_top=False) # for InceptionV3
x = GlobalAveragePooling2D()(base_model.output)
feature_ext_model = Model(inputs=base_model.inputs, outputs=x)


# Function to convert index to word
def idx_to_word(integer, index_word):
    return index_word.get(integer, None)

# Function to predict caption
def predict_caption(model, image, word_index, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_index[word] for word in in_text.split() if word in word_index]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, index_word)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Gradio interface function
def generate_caption(image):
    try:
        # Load and preprocess the image
        image = image.resize((299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract features
        # features = feature_extractor.predict(image)
        features = feature_ext_model.predict(image, verbose=0)

        # Generate caption
        caption = predict_caption(model, features, word_index, max_length)
        caption = caption.replace('startseq', '').replace('endseq', '').strip()
        return caption
    except Exception as e:
        return str(e)

# Load max_length
max_length = 74  # Use the same max_length used during training

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples") if example.endswith(".jpg")]


description = """
<h2>Important Guidelines:</h2>
<strong>Image Examples:</strong> Refer to the examples below to understand the type of images suitable for captioning.
</br>
"""

# Gradio interface setup
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label='Caption'),
    title="PicPhrase: An Image Caption Generator",
    examples=example_list,
    description=description
)

interface.launch()