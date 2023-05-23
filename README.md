# Multimodal Emotion Detection & Audio Classification
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
Can access Downloading of data into current directory, pre-processing, modeling and finally run the app using arg parse configurations for selected options.

Deployed model: It can be accessed through the pickle file in the repo.
