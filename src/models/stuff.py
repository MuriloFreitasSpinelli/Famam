def factory(music):
    piano_roll = muspy.to_pianoroll_representation(music)
    return {
        'piano_roll': piano_roll,
        'genre': music.genre,
        'artist': music.artist
    }, piano_roll  # inputs, labels

dataset = your_muspy_dataset.to_tensorflow_dataset(factory=factory)

# Model with multiple inputs
piano_roll_input = tf.keras.Input(shape=(time_steps, 128), name='piano_roll')
genre_input = tf.keras.Input(shape=(), dtype=tf.int32, name='genre') 
artist_input = tf.keras.Input(shape=(), dtype=tf.int32, name='artist')

# Embed categorical features
genre_embedding = tf.keras.layers.Embedding(num_genres, 16)(genre_input)
artist_embedding = tf.keras.layers.Embedding(num_artists, 32)(artist_input)

# Process piano roll
x = tf.keras.layers.LSTM(128)(piano_roll_input)

# Combine everything
combined = tf.keras.layers.Concatenate()([
    x, 
    tf.keras.layers.Flatten()(genre_embedding),
    tf.keras.layers.Flatten()(artist_embedding)
])

output = tf.keras.layers.Dense(128 * time_steps, activation='sigmoid')(combined)
output = tf.keras.layers.Reshape((time_steps, 128))(output)

model = tf.keras.Model(
    inputs={'piano_roll': piano_roll_input, 'genre': genre_input, 'artist': artist_input},
    outputs=output
)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(dataset.batch(32), epochs=10)