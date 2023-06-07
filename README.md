# Multimodal Emotion Recognition System to Recommend Affirmations

Abstract: Research suggests that self-affirmation energizes the brain system and stimulates neuroplasticity. Hence, automatic speech detection can detect an individual's current mood, and suggesting self-affirmation can help mitigate an individual's negative emotions and feelings. For the purpose of emotion detection inherent in a person’s speech, a multimodal network is proposed. A dual modal  network using audio and text has proven to be more effective rather than their individual classification in past research. Additionally the best network is deployed to build a custom chatbot based on GPT trained on acclaimed self-help books to recommend affirmations. The multimodal network using LSTM and GRU for text and audio respectively achieved an F1-score of 44% and a two layer Bi-GRU network with Self Attention for audio achieved a F1-score of 80%. The audio classifier was chosen to be deployed for the chatbot prototype. Through this study a different modality was used to interact with a fine-tuned pre-trained language model.   


How to Run?
1. Download code.zip and extract it
2. Open main.py and pass argument to run (--download_data download_data) to download MELD, RAVDESS, SAVEE and TESS datasets into current directory
3. Pass argument (--preprocess_ravdess preprocess_ravdess) to preprocess combined audio dataset(RAVDESS, SAVEE and TESS) and extract MFCC and MFSC features for train, val, test into current directory.
4. Pass argument --model followed by one of the choices=["Multimodal_Var1", "Multimodal_Var2", "BiGRU", "BiGRUSelfAtt", "BiGRUFuzzy"] to train, test and plot . The different models can be trained and tested. Model summary, train vs val graphs for loss and accuracy will be seen after training. After testing, classificatio report will be provided for analysis.
5. Data(docs) to train chatbot and deployed model pickle file is available in code directory. To run chatbot pass argument (--chatbot chatbot) which will recommend affirmations based on emotion detected. 

Contribution of each member:
Task	------------------------------------Member
Pre-processing of MELD	------------------Lina
Pre-processing of RAVDESS+SAVVY+TORONO---	Anisha, Faiza
Feature Extraction for MELD	--------------Anisha
Feature Extraction for Combined Audio---- Anisha,Faiza
Combined Audio Augmentation---------------Anisha, Faiza, Lina
"MELD Augmentation - Audio, Text----------Faiza, Lina
MELD Audio Models	------------------------Anisha, Faiza, Lina
MELD Text Models	------------------------Anisha, Faiza, Lina
Multimodal Models	------------------------Anisha, Lina
Combined Audio Models	--------------------Anisha, Faiza, Lina
Audio Model Deployment	------------------Anisha
Chatbot Integration	----------------------Lina
Code Conversion	--------------------------Faiza,Lina,Anisha
Slides & Report---------------------------Anisha, Faiza, Lina

About
Data Collection Procedure:
The project utilizes meld data for multimodal and three sets of audio data for audio classificatioon to detect emotions. 
To access meld data : use wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

Audio datasets:
1. RAVDESS: https://zenodo.org/record/1188976#.ZGw_B-zMLdo
2. TESS: https://tspace.library.utoronto.ca/handle/1807/24487
3. SAVEE: http://kahlan.eps.surrey.ac.uk/savee/Download.html

Pre-Processing:
Once data is in current directory the Prepreprocessing.py can be used to download features for train, test and val for both MELD and Ravdess having features for MFCC, MFSC and vectorized tokens for text into current directory.

Modeling:
Once features are extracted, the different models can be trained and tested. Model summary, train vs val graphs for loss and accuracy will be seen after training. After testing, classificatio report will be provided for analysis.

main.py:
Can access Downloading of data into current directory, pre-processing, modeling and finally run the app using arg parse configurations for selected options. To run app, API key must be updated in app.py.

Deployed model: It can be accessed through the pickle file in the repo.





