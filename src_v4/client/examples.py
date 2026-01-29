"""
Example usage of the MusicPipeline.

These examples demonstrate the complete workflow from dataset building
to music generation.
"""

from pathlib import Path


def example_full_pipeline():
    """
    Complete pipeline: build dataset, train model, generate music.
    """
    from . import MusicPipeline, PipelineConfig

    # Configure pipeline
    config = PipelineConfig(
        data_dir="data/midi",
        model_dir="models",
        output_dir="output",

        # Use REMI encoding for better polyphonic learning
        encoder_type="remi",
        resolution=24,
        max_seq_length=1024,

        # Transformer model
        model_type="transformer",
        model_name="music_transformer_v1",
        d_model=256,
        num_layers=4,
        num_heads=8,

        # Training
        batch_size=32,
        epochs=50,
        learning_rate=1e-4,

        # Generation
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    )

    # Create pipeline
    pipeline = MusicPipeline(config)

    # Step 1: Build dataset
    pipeline.build_dataset(
        midi_dir="data/midi",
        output_path="data/dataset.h5",
    )

    # Step 2: Train model
    pipeline.train()
    pipeline.save_model()

    # Step 3: Generate music
    pipeline.generate_midi(
        output_path="output/generated.mid",
        genre_id=0,
        instrument_ids=[0, 25],  # Piano and guitar
        include_drums=True,
    )

    print(pipeline.summary())


def example_load_and_generate():
    """
    Load a pre-trained model and generate music.
    """
    from . import MusicPipeline

    # Create pipeline and load model
    pipeline = MusicPipeline()
    pipeline.load_model("models/music_transformer_v1.h5")

    # List available genres
    print("Available genres:", pipeline.list_genres())

    # Generate with different settings
    for temp in [0.8, 1.0, 1.2]:
        pipeline.generate_midi(
            output_path=f"output/generated_temp{temp}.mid",
            genre_id=0,
            temperature=temp,
        )


def example_custom_generation():
    """
    Advanced generation with custom parameters.
    """
    from . import MusicPipeline, GenerationConfig

    pipeline = MusicPipeline()
    pipeline.load_model("models/music_transformer_v1.h5")

    # Generate individual tracks
    drum_track = pipeline.generate_track(
        genre_id=0,
        instrument_id=128,  # Drums
        is_drum=True,
        temperature=0.9,
    )

    piano_track = pipeline.generate_track(
        genre_id=0,
        instrument_id=0,  # Piano
        program=0,
        temperature=1.0,
    )

    bass_track = pipeline.generate_track(
        genre_id=0,
        instrument_id=33,  # Electric bass
        program=33,
        temperature=0.8,
    )

    # Combine tracks manually
    import muspy
    music = muspy.Music(
        resolution=24,
        tempos=[muspy.Tempo(time=0, qpm=120)],
        tracks=[drum_track, piano_track, bass_track],
    )
    music.write_midi("output/custom_song.mid")


def example_lstm_model():
    """
    Train and use an LSTM model instead of Transformer.
    """
    from . import MusicPipeline, PipelineConfig

    config = PipelineConfig(
        model_type="lstm",
        model_name="music_lstm_v1",
        lstm_units=512,
        d_model=256,
        epochs=30,
    )

    pipeline = MusicPipeline(config)
    pipeline.build_dataset("data/midi")
    pipeline.train()
    pipeline.save_model()


def example_event_encoder():
    """
    Use event-based encoding instead of REMI.
    """
    from . import MusicPipeline, PipelineConfig

    config = PipelineConfig(
        encoder_type="event",  # note-on/note-off/time-shift
        model_name="event_transformer",
    )

    pipeline = MusicPipeline(config)
    pipeline.build_dataset("data/midi")

    # Event encoding typically produces longer sequences
    print(f"Vocab size: {pipeline.encoder.vocab_size}")


if __name__ == "__main__":
    # Run examples
    example_full_pipeline()
