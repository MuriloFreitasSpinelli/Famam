import numpy as np
import tensorflow as tf
from src.models.lstm_model import LSTMModel

def test_lstm_save_load(tmp_path):
    vocab_size = 50
    seq_len = 8
    batch_size = 2

    model = LSTMModel(
        vocab_size=vocab_size,
        max_seq_length=seq_len,
        d_model=32,
        lstm_units=(32,),
        dropout_rate=0.0,
        recurrent_dropout=0.0,
        bidirectional=False,
    )

    x = tf.constant(
        np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
        dtype=tf.int32
    )

    y1 = model({"input_ids": x}, training=False)

    save_path = tmp_path / "test_model.keras"
    model.save(save_path)

    loaded = tf.keras.models.load_model(
        save_path,
        custom_objects={"LSTMModel": LSTMModel},
    )

    y2 = loaded({"input_ids": x}, training=False)

    assert y1.shape == y2.shape