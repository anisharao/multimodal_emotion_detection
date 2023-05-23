from keras.layers import TimeDistributed, Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, GRU, concatenate
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, SimpleRNN, Reshape, LSTM, Flatten, Layer, BatchNormalization, GRU, Attention, GlobalMaxPooling1D, Bidirectional, Embedding
import numpy as np
from keras.models import Model
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.utils import to_categorical

class Multimodal_Var1:
    """
    Class to create a Multimodal model using audio and text variant 1
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
        Load the MFCC and vectorized Text features for the meld dataset
        :return: mfcc_train, mfcc_val, mfcc_test, text_train, text_val, text_test, y_train, y_val, y_test : vectors for
        Features and labels for all training, validation, and testing
        """
        # Load the dataset
        folder = os.getcwd()
        mfcc_train = pd.read_csv(folder + '/train_mfcc_features_Meld.csv')
        mfcc_test = pd.read_csv(folder + '/test_mfcc_features_Meld.csv')
        mfcc_val = pd.read_csv(folder + '/val_mfcc_features_Meld.csv')
        print(mfcc_train)
        y_train = mfcc_train.iloc[:, -1]
        y_test = mfcc_test.iloc[:, -1]
        y_val = mfcc_val.iloc[:, -1]
        print(y_train.shape, y_train)
        # Combine all labels
        all_labels = np.concatenate([y_train, y_val, y_test])
        # Fit encoder
        le = LabelEncoder()
        le.fit(all_labels)
        # Encode the labels

        y_train = le.transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        # One-hot encoding
        print(y_train.shape, y_train[0])
        y_train = to_categorical(y_train, num_classes=7)
        y_val = to_categorical(y_val, num_classes=7)
        y_test = to_categorical(y_test, num_classes=7)

        text_train = pd.read_csv(folder + '/train_text_Meld.csv')
        text_test = pd.read_csv(folder + '/test_text_Meld.csv')
        text_val = pd.read_csv(folder + '/val_text_Meld.csv')

        mfcc_train = np.array(mfcc_train.iloc[:, :-1])
        mfcc_test = np.array(mfcc_test.iloc[:, :-1])
        mfcc_val = np.array(mfcc_val.iloc[:, :-1])
        # Reshape the input data to have a time axis

        mfcc_train = np.expand_dims(mfcc_train, axis=-1)
        mfcc_val = np.expand_dims(mfcc_val, axis=-1)
        mfcc_test = np.expand_dims(mfcc_test, axis=-1)

        text_train = np.array(text_train.iloc[:, :-1])
        text_test = np.array(text_test.iloc[:, :-1])
        text_val = np.array(text_val.iloc[:, :-1])


        text_train = np.expand_dims(text_train, axis=2)
        text_val = np.expand_dims(text_val, axis=2)
        text_test = np.expand_dims(text_test, axis=2)
        print(mfcc_train.shape)
        #self.NUM_MFCC = mfcc_train.shape[1]

        return (mfcc_train, text_train, y_train), (mfcc_val, text_val, y_val), (mfcc_test, text_test, y_test)

    def build_model(self):
        """
        Function to build the Multimodal model first variant
        :return: the Multimodal model first variant
        """
        # Define the model architecture
        # Input layer for the MFCCs (GRU component)
        input_mfcc = Input(shape=(self.NUM_MFCC,))
        x = Embedding(self.max_features, self.embedding_dims)(input_mfcc)
        x = GRU(units=64, activation='relu', return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(self.dropout_rate)(x)

        # Input layer for the text (LSTM component)
        input_text = Input(shape=(self.max_len,))
        y = Embedding(self.max_features, self.embedding_dims)(input_text)
        y = LSTM(self.hidden_dims, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate, return_sequences=True)(y)
        y = LSTM(self.hidden_dims, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate, return_sequences=True)(y)
        y = BatchNormalization()(y)
        y = SeqSelfAttention()(y)
        y = GlobalMaxPooling1D()(y)
        y = Dropout(self.dropout_rate)(y)

        # Concatenate the outputs from the two components
        combined = concatenate([x, y])

        # Output layer for the classification
        output = Dense(units=7, activation='softmax')(combined)

        # Define the model
        model = Model(inputs=[input_mfcc, input_text], outputs=output)
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def plot_loss_acc_graph(self, history):
        """
        Method to plot the loss and accuracy graphs of the trained model
        :param history: a dataframe that saved the history of the model while training
        :return: Displays Plots of loss and accuracy
        """
        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation loss for network')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs Validation accuracy for network')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    def train(self):
        """
        Method to train the Multimodal model first variant
        :return: prints the model summary and plot loss and accuracy
        """
        # Load the dataset
        (mfcc_train, text_train, y_train), (mfcc_val, text_val, y_val), (mfcc_test, text_test, y_test) = self.load_dataset()

        print("Shapes")
        print(mfcc_train.shape,text_train.shape,y_train.shape)

        # Build the model
        self.model = self.build_model()

        # Train the model
        history = self.model.fit([mfcc_train, text_train], y_train,
                                 validation_data=([mfcc_val, text_val], y_val),
                                 epochs=10, batch_size=32)
        print(self.model.summary())

        self.plot_loss_acc_graph(history)

    def predict(self):
        """
        Method that calculates the predicted values and compares them to real test labels
        :return: Prints test accuracy and F1-score
        """
        # Load the dataset
        _, _, (mfcc_test, text_test, y_test) = self.load_dataset()

        # Evaluate model with the test set
        eval_score = self.model.evaluate([mfcc_test, text_test], y_test)
        print('Test loss:', eval_score[0])
        print('Test accuracy:', eval_score[1])

        # Make predictions on the test set
        y_pred = self.model.predict([mfcc_test, text_test])
        y_pred = np.argmax(y_pred, axis=1)

        # Compute the F1 score
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
        print("F1 score:", f1)
        # Classification report
        print(classification_report(np.argmax(y_test, axis=1), y_pred))
