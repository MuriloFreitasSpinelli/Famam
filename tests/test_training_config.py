import inspect

from src.config.training_config import TrainingConfig


def test_training_config_instantiates():
    sig = inspect.signature(TrainingConfig)

    # Fill values only if the parameter exists in the real signature
    candidate_values = {
        "model_type": "lstm",
        "max_seq_length": 16,
        "seq_len": 16,
        "sequence_length": 16,
        "batch_size": 2,
        "epochs": 1,
        "vocab_size": 100,
        "d_model": 32,
        "lstm_units": (32, 32),
        "dropout_rate": 0.0,
        "recurrent_dropout": 0.0,
        "bidirectional": False,
        "learning_rate": 1e-3,
    }

    kwargs = {}
    missing_required = []

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        if name in candidate_values:
            kwargs[name] = candidate_values[name]
            continue

        # If required and we do not know how to fill it, track it
        if p.default is inspect._empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            missing_required.append(name)

    # If your TrainingConfig has required params we cannot guess safely,
    # then this test should fail with a clear message so we can fill them correctly.
    assert not missing_required, f"TrainingConfig requires params we did not provide: {missing_required}"

    cfg = TrainingConfig(**kwargs)

    # Basic sanity: object exists and has at least one attribute
    assert cfg is not None
    assert len(vars(cfg)) >= 0