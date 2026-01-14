##TODO get data set from the saved model, present in the class saved model,
# class in the model generation, all of the configs, also in saved model
# get the music representation from the tensorflow data set config,
# then get the tensor type from the tensor flow data set config,
#start parsing the data, list of strings
# first item in the list of strings is genre
# second instrument
# check model type if it wants model/instrument
# if it has arguments then it is incorrect
# type is genre and only genre -> try to get the genre id from the data set vocabulary (enchanced music dataset)
# map, if it doesnt find the string then dont generate -> error
# same thing is instrument, if it doesnt find the instrument from the enchanced music dataset, send error
# genre and instrument is something u can receive as arguments, same thing
# call generate on the saved model with the data
# return an array of something (bytes)
# Use the Muspy library
# Call read?
# Call it back to the representation from the tensorflow data set
# mapping to events etc
# transform new data to MIDI
# See the documentation from the MusPy (skip all the testing n stuff)
# write a method that takes in a string path & it is going to go to the path and load the save path (does it exist?)
# saved model class

#claude
#generate some tests