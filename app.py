import pickle
import os
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import librosa
import numpy as np

# Load the pickle model
folder = os.getcwd()
model_path = folder +"/biGRU_audio_model_deploy.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

os.environ["OPENAI_API_KEY"] = 'ADD_API_KEY' 
NUM_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050
DOWN_SAMPLE_RATE = 16000
def extract_features(data):
    """
    Method to extract Mel Frequency features from an audio file
    :param data (wav): audio file
    :return: Mel Spectrogram features
    """
    # mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feature = np.mean(spectrogram.T, axis = 0 )
    # print(feature)
    return feature
def get_class_from_value(value, labels):
    """
    Get the emotion class from a given value
    :param value:
    :param labels:
    :return: label corresponding to the value
    """
    for label, class_value in labels.items():
        if class_value == value:
            return label

def construct_index(directory_path):
    """

    :param directory_path:
    :return:
    """
    # Refrenced from : https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api/

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index
def chatbot(audio_file):
    """

    :param audio_file:
    :return:
    """

    print(audio_file)
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_file)
    mfsc=[]
    # Extract MFSC features from the audio file
    mfsc_features = extract_features(audio)  # Implement this function
    mfsc.append(mfsc_features)
    mfsc = np.expand_dims(mfsc, axis=2)
    # Use the model to predict the emotion probabilities
    emotion_probs = model.predict(mfsc)[0]

    # Get the predicted emotion index with the highest probability
    predicted_emotion_index = np.argmax(emotion_probs)
    labels={'neutral': 0, 'happiness': 1, 'sadness': 2, 'fear': 3, 'anger': 4}
    predicted_emotion = get_class_from_value(predicted_emotion_index, labels)

    # Use the predicted emotion as the prompt for the chatbot
    prompt = "I' am feeling "+predicted_emotion+". Give me some suggestions to handle this emotion."

    index = construct_index(folder+"/docs")
    response = index.query(prompt, response_mode="compact")
    res= "The inherent emotion in the audio is, "+predicted_emotion+".\n"+response.response
    return res

inputs=gr.inputs.File(label="Upload an audio file")
print(inputs)
iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Audio(label="Speak", type="filepath"),
                     outputs="text",
                     title="Peacify")

iface.launch(share=True)
