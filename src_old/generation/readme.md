TODO:

make a script that can generate the songs.

with a method that takes in an enhanced_music_dataset and a saved_model. Make it so that you can generate using the model. For this to happen
you need to input the correct mapped values when calling the predict method on tensorflow. Example: a model trained on a dataset with a certain datasetconfig can only
be generated with the tensors saved in the dataset. So a tensor with the format {"music": ?, "genre": ?} cannot be used to predict one with format {"music": ?}. Also even if they do have the same format many different datasets have different mappings, due to a difference in genre, so always map the values asked for from there. SO you get genre = "rock" from dataset.vocab.get_genre() -> returns the int related to that genre, that the model undestands.

FOr this you need to use the data available in the enhanced_dataset class, also reference the tensorflow_dataset_config created from the dataset to see which representation you need to switch back from. all this data about the configs and used dataset will be present on the saved_model_class.

---

Additional notes:
- get data set from the saved model, present in the class saved model,
- class in the model generation, all of the configs, also in saved model
- get the music representation from the tensorflow data set config,
- then get the tensor type from the tensor flow data set config,
- start parsing the data, list of strings
- first item in the list of strings is genre
- second instrument
- check model type if it wants model/instrument
- if it has arguments then it is incorrect
- type is genre and only genre -> try to get the genre id from the data set vocabulary (enchanced music dataset)
- map, if it doesnt find the string then dont generate -> error
- same thing is instrument, if it doesnt find the instrument from the enchanced music dataset, send error
- genre and instrument is something u can receive as arguments, same thing
- call generate on the saved model with the data
- return an array of something (bytes)
- Use the Muspy library
- Call read?
- Call it back to the representation from the tensorflow data set
- mapping to events etc
- transform new data to MIDI
- See the documentation from the MusPy (skip all the testing n stuff)
- write a method that takes in a string path & it is going to go to the path and load the save path (does it exist?)
- saved model class

claude: generate some tests
