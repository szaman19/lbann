import urllib.request
import os 
import os.path 
import zipfile 
import sdf

data_dir = os.path.dirname(os.path.realpath(__file__))
def download_data():
    """ Download QM9 data, if needed. 
    

    Data files are downloaded from http:"""

    url = "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip"
    print("Downloading QM9 data")
    data_file_name = os.path.join(data_dir, "QM9")
    compressed_file_name = data_file_name + ".zip"
    
    # Assume data does not exist 
    
    urllib.request.urlretrieve(url, filename=compressed_file_name)
    with zipfile.ZipFile("QM9.zip", "r") as zip_file:
        zip_file.extractall(data_file_name)

def extract_data():
    


if __name__ == '__main__':
    #download_data()
    extract_data()
