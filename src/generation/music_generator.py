import tensorflow as tf
from typing import List

from src.data.dataset_vocabulary import DatasetVocabulary


def generate_music_from_data(savedmodel, data):
    outputs = []
    for sample in data:
        prediction = savedmodel(sample, training=False)
        outputs.append(prediction)
    return tf.concat(outputs, axis=0)


def parse_genre_data(data : List[str], library : DatasetVocabulary):
    genre = data[0]
    genre_id = library.get_genre_id(genre)
    # if (library.genre_to_id)
    return genre_id
        #       First item of the list is the genre

def parse_instrument_data(data : List[str], library : DatasetVocabulary):
    instrument = data[1]
    instrument_id = library.get_instrument_id(instrument)
    return instrument_id
