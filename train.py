import os
from audio.read_audio import read_audio_dataset
from dataset.dataset_reader import read_file

if __name__ == '__main__':
    files = read_file('train_data.csv', root_dir='timit')

    dataset = read_audio_dataset(files)

    print(dataset)