


for midi in passed_filter(FilterCOnfig):
    music = muspy.Music.read(path to midi)
    for track in music.tracks:
        track.program = isntrument_vocab(track.program)
    metadata = {genre: string, artist: string, instruments: string}
    enhancedMusic = EnhancedMusic(music, metadata)

for music in all_enhasedmusic:
    #preprocess

dataset = enhanceddataset(preprocessed_musics)
dataset.build_vocabulary()
dataset.save()

