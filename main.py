import argparse
import os
import Preprocessing
import Multimodal_Var1
import Multimodal_Var2
import BiGRU
import BiGRUSelfAtt
import BiGRUFuzzy
import Download_Data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a multimodal emotion detection model')
    parser.add_argument("--download_meld_data", help="Downloading meld data")
    parser.add_argument("--preprocess_ravdess",help="Clean, augment and Extract MFCC and MFSC Features for Ravdes and Tess Dataset")
    parser.add_argument("--preprocess_meld", help="Clean, augment and Extract MFCC and MFSC Features for MELD Dataset")
    parser.add_argument("--wav2vec_meld", help="Clean, augment and Extract Wav2Vec Features for MELD Dataset")
    parser.add_argument("--model", help="Train the Model", choices=["Multimodal_Var1", "Multimodal_Var2", "BiGRU", "BiGRUSelfAtt", "BiGRUFuzzy"])
    parser.add_argument("--chatbot", help="Recommend Affirmations")

    args = parser.parse_args()
    folder = os.getcwd()

    if args.download_meld_data == "download_meld_data":
        """
        Dowmload the MELD dataset 
        """
        download_meld = Download_Data.Download_Data()
        # download_meld.download_meld_csv_data()
        # download_meld.download_meld_audio_data()
        download_meld.download_combined_audio()


    if args.preprocess_meld == "preprocess_meld":
        """
        preprocess the MELD dataset 
        """
        preprocess = Preprocessing.Preprocessing()
        preprocess.preprocess_meld()

    if args.preprocess_ravdess == "preprocess_ravdess":
        """
        preprocess the RAVDESS-TESS-SAVEE dataset 
        """
        preprocess = Preprocessing.Preprocessing()
        preprocess.preprocess_ravdess()

    if args.model == "Multimodal_Var1":
        """
        Train the MultiModal model first variant
        """
        # physical_devices = tf.config.list_physical_devices('GPU')
        # print(physical_devices)
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = Multimodal_Var1.Multimodal_Var1(max_features=20000, max_len=400, embedding_dims=32, hidden_dims=64, dropout_rate=0.5)
        model.train()
        model.predict()

    if args.model == "Multimodal_Var2":
        """
        Train the MultiModal model second variant
        """
        model=Multimodal_Var2.Multimodal_Var2(max_features=20000, max_len=400, embedding_dims=32, hidden_dims=64, dropout_rate=0.2)
        model.train()
        model.predict()

    if args.model == "BiGRU":
        """
        Train the BiGRU model
        """
        model=BiGRU.BiGRU()
        model.train()
        model.predict()
    if args.model == "BiGRUSelfAtt":
        """
        Train the BiGRU model with self-Attention
        """
        model=BiGRUSelfAtt.BiGRUSelfAtt()
        model.train()
        model.predict()
    if args.model == "BiGRUFuzzy":
        """
       Train the BiGRU model with Fuzzy Attention
       """
        model=BiGRUFuzzy.BiGRUFuzzy()
        model.train()
        model.predict()
    if args.chatbot == "chatbot":
        """
        Run the ChatBot
        """
        # execfile('app.py')
        exec(open('app.py').read())
