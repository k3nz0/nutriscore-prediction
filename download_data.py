import zipfile
from download import download

print("Donwloading data.zip (191M), this could take a while.")

url = "https://www.dropbox.com/s/dayh1zd5qm2xufu/data.zip?dl=0"
path = "./data/data.zip"

path = download(url=url, path=path, progressbar=True, replace=True)

print("Extracting data.zip in ./data/")

with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall('./data/')

