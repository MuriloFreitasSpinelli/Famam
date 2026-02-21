from src.config.training_config import TrainingConfig
from src.config.music_dataset_config import MusicDatasetConfig


def test_config_objects_construct():
    t = TrainingConfig()
    d = MusicDatasetConfig()
    assert t is not None
    assert d is not None