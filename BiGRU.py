from keras.layers import TimeDistributed, Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, GRU, concatenate
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, SimpleRNN, Reshape, LSTM,Flatten, Layer,BatchNormalization, GRU, Attention,GlobalMaxPooling1D,Bidirectional,Embedding
import numpy as np
from keras.models import Model
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.utils import to_categorical

# Load the dataset
folder = os.getcwd()
mfcc=pd.read_csv(folder+"/train_mfsc_features_ravdes.csv")
mfcc_test=pd.read_csv(folder+"/test_mfsc_features_ravdes.csv")
mfcc_val=pd.read_csv(folder+"/valid_mfsc_features_ravdes.csv")
X_train = mfcc.iloc[:, :128]  # Select first 128 columns as features
y_train = mfcc.iloc[:, -1]   # Select last column as labels

X_val = mfcc_val.iloc[:, :128]  # Select first 128 columns as features
y_val = mfcc_val.iloc[:, -1]   # Select last column as labels

X_test = mfcc_test.iloc[:, :128]  # Select first 128 columns as features
y_test = mfcc_test.iloc[:, -1]

#Combine all labels
all_labels = np.concatenate([y_train, y_val, y_test])

#Fit encoder
le = LabelEncoder()
le.fit(all_labels)

# Encode the labels
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

#one hot encoding
y_train = to_categorical(y_train, num_classes=5)
y_val = to_categorical(y_val, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

X_train = np.expand_dims(X_train, axis = 2)
X_test = np.expand_dims(X_test, axis = 2)
X_val = np.expand_dims(X_val, axis = 2)

class BiGRU:
    def __init__(self):

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(GRU(64, return_sequences=True, input_shape=(128, 1))))
        model.add(Bidirectional(GRU(64)))
        model.add(Dropout(.3))
        model.add(Dense(units=5, activation='softmax'))

        # Compile the model/drive/1YybVX1Nia14Uucd4lRhrPDuH_2jcXNkv
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        return model

    def plot_loss_acc_graph(history):
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

        # Train the model
        history=self.model.fit(X_train, y_train, epochs=40, batch_size=80, validation_data=(X_val, y_val))
        print(self.model.summary())

        self.plot_loss_acc_graph(history)

    def predict(self):
        # Evaluate model with test set
        eval_score = self.model.evaluate(X_test, y_test)
        print('Test loss:', eval_score[0])
        print('Test accuracy:', eval_score[1])

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # Compute the F1 score
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
        print("F1 score:", f1)
        # classification report
        print(classification_report(np.argmax(y_test, axis=1), y_pred))
