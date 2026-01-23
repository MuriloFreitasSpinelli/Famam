import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, List

from src.model_training.configs.model_training_config import ModelTrainingConfig


@dataclass
class ModelTuningConfig:
    """
    Configuration for hyperparameter tuning of music generation models.

    Supports multiple tuning strategies:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling of parameter space
    - Bayesian Optimization: Sequential model-based optimization

    Works with ModelTrainingConfig to define base training parameters
    and parameter grids for tuning.
    """

    VALID_TUNING_METHODS: ClassVar[set[str]] = {
        'grid_search', 'random_search', 'bayesian'
    }
    VALID_SCORING_METRICS: ClassVar[set[str]] = {
        'neg_mean_squared_error', 'neg_mean_absolute_error',
        'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
    }

    # ============ Cross-Validation Settings ============
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: Optional[int] = 42

    # ============ Hyperparameter Tuning ============
    use_hyperparameter_tuning: bool = False
    tuning_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'

    # Parameter grid for tuning (if None, will use default grid)
    param_grid: Optional[Dict[str, List[Any]]] = None

    # Random search specific
    n_iter: int = 10  # Number of parameter settings sampled

    # Bayesian optimization specific
    bayesian_n_calls: int = 50
    bayesian_n_initial_points: int = 10

    # Tuning validation
    tuning_cv_folds: int = 3
    tuning_scoring: str = 'neg_mean_squared_error'
    tuning_n_jobs: int = 1  # Number of parallel jobs (-1 for all CPUs, 1 for safe cluster use)
    tuning_verbose: int = 2
    tuning_batch_size: int = 16  # Batch size for tuning (smaller to avoid OOM)
    tuning_epochs: int = 10  # Epochs per trial during tuning

    # ============ Output Settings ============
    output_dir: str = './models'

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate tuning method
        if self.use_hyperparameter_tuning:
            if self.tuning_method not in self.VALID_TUNING_METHODS:
                raise ValueError(
                    f"Invalid tuning_method '{self.tuning_method}'. "
                    f"Must be one of: {', '.join(sorted(self.VALID_TUNING_METHODS))}"
                )

        # Validate CV folds
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got {self.cv_folds}")

        if self.tuning_cv_folds < 2:
            raise ValueError(f"tuning_cv_folds must be at least 2, got {self.tuning_cv_folds}")

        # Validate n_iter for random search
        if self.tuning_method == 'random_search' and self.n_iter < 1:
            raise ValueError(f"n_iter must be positive, got {self.n_iter}")

        # Create default param_grid if using tuning but no grid provided
        if self.use_hyperparameter_tuning and self.param_grid is None:
            self.param_grid = self._create_default_param_grid()

    def _create_default_param_grid(self) -> Dict[str, List[Any]]:
        """Create a default parameter grid for hyperparameter tuning."""
        return {
            'lstm_units': [[64], [128], [128, 64], [256, 128]],
            'dense_units': [[32], [64], [64, 32]],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'rmsprop'],
        }

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Tuning configuration saved to: {output_path}")

    @classmethod
    def load(cls, path: str) -> 'ModelTuningConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def save_tuning_results(self, search_cv, experiment_name: str) -> str:
        """
        Save hyperparameter tuning results to JSON file.

        Args:
            search_cv: Fitted GridSearchCV, RandomizedSearchCV, or BayesSearchCV object
            experiment_name: Name for this tuning experiment

        Returns:
            Path to the saved results file
        """
        results_path = Path(self.output_dir) / 'tuning' / f'{experiment_name}.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract results from the search object
        results = {
            'experiment_name': experiment_name,
            'tuning_method': self.tuning_method,
            'best_params': search_cv.best_params_,
            'best_score': float(search_cv.best_score_),
            'cv_folds': self.tuning_cv_folds,
            'scoring': self.tuning_scoring,
            'cv_results': {
                'mean_test_score': [float(x) for x in search_cv.cv_results_['mean_test_score']],
                'std_test_score': [float(x) for x in search_cv.cv_results_['std_test_score']],
                'mean_train_score': [float(x) for x in search_cv.cv_results_.get('mean_train_score', [])],
                'params': [
                    {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in p.items()}
                    for p in search_cv.cv_results_['params']
                ],
                'rank_test_score': [int(x) for x in search_cv.cv_results_['rank_test_score']],
            },
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Tuning results saved to: {results_path}")
        return str(results_path)

    @classmethod
    def from_tuning_results(
        cls,
        results_path: str,
        model_name: str,
        **overrides
    ) -> ModelTrainingConfig:
        """
        Create a ModelTrainingConfig using best parameters from tuning results.

        Args:
            results_path: Path to the tuning results JSON file
            model_name: Name for the new model
            **overrides: Additional parameters to override (e.g., epochs=200)

        Returns:
            ModelTrainingConfig instance with best tuned parameters
        """
        with open(results_path, 'r') as f:
            results = json.load(f)

        best = results['best_params']

        # Map tuning params to config params
        config_kwargs = {
            'model_name': model_name,
            # From best params (tuned)
            'lstm_units': best.get('lstm_units', [128, 64]),
            'dense_units': best.get('dense_units', [64, 32]),
            'dropout_rate': best.get('dropout_rate', 0.2),
            'recurrent_dropout': best.get('recurrent_dropout', 0.1),
            'learning_rate': best.get('learning_rate', 0.001),
            'batch_size': best.get('batch_size', 32),
            'optimizer': best.get('optimizer', 'adam'),
            'bidirectional': best.get('bidirectional', False),
            'l1_reg': best.get('l1_reg', 0.0),
            'l2_reg': best.get('l2_reg', 0.0),
        }

        # Apply any overrides
        config_kwargs.update(overrides)

        config = ModelTrainingConfig(**config_kwargs)

        print(f"Created config from tuning results:")
        print(f"  Source: {results_path}")
        print(f"  Best score: {results['best_score']:.4f}")
        print(f"  Best params: {best}")

        return config

    @staticmethod
    def list_tuning_results(output_dir: str = './models') -> List[Dict[str, Any]]:
        """
        List all saved tuning results.

        Args:
            output_dir: Base output directory

        Returns:
            List of tuning result summaries
        """
        tuning_dir = Path(output_dir) / 'tuning'
        if not tuning_dir.exists():
            return []

        results = []
        for path in tuning_dir.glob('*.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            results.append({
                'path': str(path),
                'experiment_name': data.get('experiment_name', path.stem),
                'best_score': data.get('best_score'),
                'best_params': data.get('best_params'),
                'tuning_method': data.get('tuning_method'),
            })

        # Sort by best score (descending for negative scores like neg_mse)
        results.sort(key=lambda x: x.get('best_score', float('-inf')), reverse=True)
        return results

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        summary_lines = [
            "=" * 70,
            "Model Tuning Configuration Summary",
            "=" * 70,
            "",
            "Cross-Validation:",
            f"  Enabled: {self.use_cross_validation}",
            f"  Folds: {self.cv_folds}",
            f"  Shuffle: {self.cv_shuffle}",
            f"  Random State: {self.cv_random_state}",
            "",
            "Hyperparameter Tuning:",
            f"  Enabled: {self.use_hyperparameter_tuning}",
            f"  Method: {self.tuning_method}",
            f"  CV Folds: {self.tuning_cv_folds}",
            f"  Scoring: {self.tuning_scoring}",
            f"  N Jobs: {self.tuning_n_jobs}",
        ]

        if self.tuning_method == 'random_search':
            summary_lines.append(f"  N Iterations: {self.n_iter}")
        elif self.tuning_method == 'bayesian':
            summary_lines.extend([
                f"  Bayesian Calls: {self.bayesian_n_calls}",
                f"  Initial Points: {self.bayesian_n_initial_points}",
            ])

        if self.param_grid:
            summary_lines.extend([
                "",
                "Parameter Grid:",
            ])
            for param, values in self.param_grid.items():
                summary_lines.append(f"  {param}: {values}")

        summary_lines.extend([
            "",
            "Output:",
            f"  Output Directory: {self.output_dir}",
            "=" * 70,
        ])

        return "\n".join(summary_lines)
