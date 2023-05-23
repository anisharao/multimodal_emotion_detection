import gdown
from zipfile import ZipFile
from fsspec.compression import unzip

class Download_Data:
    """
    Class to download data from their respective locations
    """
    def download_meld_csv_data(self):
        """
        Method to download the MELD CSV files
        :return: downloads the MELD CSV files
        """
        output_files = ['dev_sent_emo', 'test_sent_emo', 'train_sent_emo']
        url_list = ['https://drive.google.com/uc?id=132cG5WA61CAF211GA3r5Npp2PopZxVl6',
                    'https://drive.google.com/uc?id=1IKHyPIbhTKyDnu5EayYgSTXBmCXZPonw',
                    'https://drive.google.com/uc?id=1lw8rtqmLBxPi9w3l10TPQMzvgH3HRAQW']
        for file,url in zip(output_files,url_list):
            gdown.download(url, file+".csv", quiet=False)

    def download_meld_audio_data(self):
        """
        Method to download the MELD zipped audio files
        :return: Extract all MELD audios
        """
        output_files = ['audio_train.zip', 'audio_test.zip', 'audio_val.zip']
        url_list = ['https://drive.google.com/uc?id=115uhYSEMBkQWXHT3zz1o8PH7gYDJrVQd',
                    'https://drive.google.com/uc?id=12_T3ONdE5LhEKr2SCIJd18k-SRXUMHgs',
                    'https://drive.google.com/uc?id=1GiDVwTdxVr1cKKPsbfzYh5tseEFbnBd6']
        for file,url in zip(output_files,url_list):
            gdown.download(url, file, quiet=False)
            with ZipFile(file, 'r') as f:
                f.extractall()

    def download_combined_audio(self):
        """
        Method to download the RAVDES, TESS. and SAVEE files
        :return: Extract all audios
        """
        output_files = ['savee.zip', 'tess.zip', 'ravdess.zip']
        url_list = ['https://drive.google.com/uc?id=18xWMOIjcMzLau_PnWpLtNUOKjRAct0I5',
                    'https://drive.google.com/uc?id=11dFO4ZYEOYlLPznksuLONBYDGnz_KeHW',
                    'https://drive.google.com/uc?id=1qHuJPLyCD3RuUQNiWpVMv4XC4M6Qdu9w',]
        for file, url in zip(output_files, url_list):
            gdown.download(url, file , quiet=False)
            with ZipFile(file, 'r') as f:
                f.extractall()
