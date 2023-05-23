import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    TimeDistributed,
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    GRU,
    concatenate,
    Bidirectional,
)
from tensorflow.keras.models import Model
from keras_self_attention import SeqSelfAttention
import tensorflow as tf

class Multimodal_Var2:
    """
    Class to create a Multimodal model using audio and text variant 2
    """
    def __init__(self, max_features, max_len, embedding_dims, hidden_dims, dropout_rate):
        """
        Constructs the model
        :param max_features: maximum feature for embedding
        :param max_len: maximun length of text
        :param embedding_dims: Embedding dimension before the
        :param hidden_dims: number of hidden units
        :param dropout_rate: Dropout rate for the dropout layer
        """
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.NUM_MFCC = 20
        self.model = self.build_model()

    def load_dataset(self):
        """
        Load the MFCC features and vectorized Text for the meld dataset
        :return: mfcc_train, mfcc_val, mfcc_test, text_train, text_val, text_test, y_train, y_val, y_test : vectors for
        Features and labels for all training, validation, and testing
        """
        folder = os.getcwd()
        mfcc_train = pd.read_csv(folder + '/train_mfcc_features_Meld.csv')
        mfcc_test = pd.read_csv(folder + '/test_mfcc_features_Meld.csv')
        mfcc_val = pd.read_csv(folder + '/val_mfcc_features_Meld.csv')

        y_train = mfcc_train.iloc[:, -1]
        y_test = mfcc_test.iloc[:, -1]
        y_val = mfcc_val.iloc[:, -1]

        all_labels = np.concatenate([y_train, y_val, y_test])

        le = LabelEncoder()
        le.fit(all_labels)

        y_train = le.transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        y_train = to_categorical(y_train, num_classes=7)
        y_val = to_categorical(y_val, num_classes=7)
        y_test = to_categorical(y_test, num_classes=7)

        text_train = pd.read_csv(folder + '/train_text_Meld.csv')
        text_test = pd.read_csv(folder + '/test_text_Meld.csv')
        text_val = pd.read_csv(folder + '/val_text_Meld.csv')

        self.NUM_MFCC = mfcc_train.shape[1]

        mfcc_train = np.array(mfcc_train.iloc[:, :-1])
        mfcc_test = np.array(mfcc_test.iloc[:, :-1])
        mfcc_val = np.array(mfcc_val.iloc[:, :-1])

        mfcc_train = np.expand_dims(mfcc_train, axis=-1)
        mfcc_val = np.expand_dims(mfcc_val, axis=-1)
        mfcc_test = np.expand_dims(mfcc_test, axis=-1)

        text_train = np.array(text_train.iloc[:, :-1])
        text_test = np.array(text_test.iloc[:, :-1])
        text_val = np.array(text_val.iloc[:, :-1])

        text_train = np.expand_dims(text_train, axis=2)
        text_val = np.expand_dims(text_val, axis=2)
        text_test = np.expand_dims(text_test, axis=2)

        return mfcc_train, mfcc_val, mfcc_test, text_train, text_val, text_test, y_train, y_val, y_test

    def build_model(self):
        """
        Function to build the Multimodal model second variant
        :return: the Multimodal model second variant
        """
        audio_input = Input(shape=(self.NUM_MFCC,))
        text_input = Input(shape=(self.max_len,))

        text_embedded = Embedding(input_dim=self.max_features, output_dim=64)(text_input)
        gru_layer1 = Bidirectional(GRU(units=256, dropout=.2, activation='tanh', return_sequences=True))(text_embedded)
        gru_layer2 = Bidirectional(GRU(units=256, dropout=.2, activation='tanh', return_sequences=True))(gru_layer1)
        att = SeqSelfAttention()(gru_layer2)
        dropout_layer_text = Dropout(.1)(att)
        pool_layer = (tf.keras.layers.GlobalAveragePooling1D())(dropout_layer_text)
        dropout_layer_text = Dropout(.1)(pool_layer)

        x = Embedding(input_dim=self.max_features, output_dim=64)(audio_input)
        gru = GRU(units=128, return_sequences=True)(x)
        x = GRU(units=64, activation='relu', return_sequences=True)(gru)
        att = SeqSelfAttention()(x)
        x = GlobalMaxPooling1D()(att)
        dropout_layer_audio = Dropout(.1)(x)

        concat = concatenate([dropout_layer_text, dropout_layer_audio])
        fc1 = Dense(256, activation='relu')(concat)
        output = Dense(7, activation='softmax')(fc1)

        model = Model(inputs=[audio_input, text_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def plot_loss_acc_graph(self, history):
        """
        Method to plot the loss and accuracy graphs of the trained model
        :param history: a dataframe that saved the history of the model while training
        :return: Displays Plots of loss and accuracy
        """
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation loss for network')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs Validation accuracy for network')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    def train(self):
        """
        Method to train the Multimodal model second variant
        :return: prints the model summary and plot loss and accuracy
        """
        (
            mfcc_train,
            mfcc_val,
            mfcc_test,
            text_train,
            text_val,
            text_test,
            y_train,
            y_val,
            y_test,
        ) = self.load_dataset()

        self.model = self.build_model()

        history = self.model.fit(
            [mfcc_train, text_train],
            y_train,
            validation_data=([mfcc_val, text_val], y_val),
            epochs=10,
            batch_size=32
        )

        print(self.model.summary())
        self.plot_loss_acc_graph(history)

    def predict(self):
        """
        Method that calculates the predicted values and compares them to real test labels
        :return: Prints test accuracy and F1-score
        """
        (
            mfcc_train,
            mfcc_val,
            mfcc_test,
            text_train,
            text_val,
            text_test,
            y_train,
            y_val,
            y_test,
        ) = self.load_dataset()

        eval_score = self.model.evaluate([mfcc_test, text_test], y_test)
        print('Test loss:', eval_score[0])
        print('Test accuracy:', eval_score[1])

        y_pred = self.model.predict([mfcc_test, text_test])
        y_pred = np.argmax(y_pred, axis=1)

        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
        print("F1 score:", f1)

        print(classification_report(np.argmax(y_test, axis=1), y_pred))
