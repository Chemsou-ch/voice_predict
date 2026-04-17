import os
import tarfile
import urllib.request
import shutil

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_NAME = "speech_commands.tar.gz"
EXTRACT_FOLDER = "speech_commands_full"
TARGET_FOLDER = "dataset"
commands = ["yes", "no", "stop", "go"]

def download_file():
    if not os.path.exists(ARCHIVE_NAME):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, ARCHIVE_NAME)

def extract_archive():
    if not os.path.exists(EXTRACT_FOLDER):
        print("Extracting...")
        with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
            tar.extractall(EXTRACT_FOLDER)

def prepare_dataset():
    for cmd in commands:
        src = os.path.join(EXTRACT_FOLDER, cmd)
        dst = os.path.join(TARGET_FOLDER, cmd)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("Copied:", cmd)

download_file()
extract_archive()
prepare_dataset()
print("Dataset ready!")