TODO:

make a script that can generate the songs.

with a method that takes in an enhanced_music_dataset and a saved_model. Make it so that you can generate using the model. For this to happen
you need to input the correct mapped values when calling the predict method on tensorflow. Example: a model trained on a dataset with a certain datasetconfig can only
be generated with the tensors saved in the dataset. So a tensor with the format {"music": ?, "genre": ?} cannot be used to predict one with format {"music": ?}. Also even if they do have the same format many different datasets have different mappings, due to a difference in genre, so always map the values asked for from there. SO you get genre = "rock" from dataset.vocab.get_genre() -> returns the int related to that genre, that the model undestands.

FOr this you need to use the data available in the enhanced_dataset class, also reference the tensorflow_dataset_config created from the dataset to see which representation you need to switch back from. all this data about the configs and used dataset will be present on the saved_model_class.