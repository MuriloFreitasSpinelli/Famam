"""
TensorFlow distribution strategy factory for distributed training.

Provides factory functions to create appropriate tf.distribute.Strategy
instances for various training scenarios:
- Single GPU (no strategy or OneDeviceStrategy)
- Multi-GPU on single node (MirroredStrategy)
- Multi-node distributed (MultiWorkerMirroredStrategy)
- TPU (TPUStrategy)
"""

import os
import json
from typing import Optional, List, Dict, Any

import tensorflow as tf  # type: ignore


class DistributionStrategyFactory:
    """
    Factory class for creating TensorFlow distribution strategies.

    Supports:
    - 'none': Default strategy (single device)
    - 'mirrored': MirroredStrategy for multi-GPU on single node
    - 'multi_worker_mirrored': MultiWorkerMirroredStrategy for multi-node
    - 'tpu': TPUStrategy for TPU training
    """

    VALID_STRATEGIES = {'none', 'mirrored', 'multi_worker_mirrored', 'tpu'}

    @staticmethod
    def create_strategy(
        strategy_name: str = 'none',
        devices: Optional[List[str]] = None,
        cross_device_ops: str = 'nccl',
        tpu_address: Optional[str] = None,
    ) -> tf.distribute.Strategy:
        """
        Create a distribution strategy based on the specified type.

        Args:
            strategy_name: Strategy type ('none', 'mirrored', 'multi_worker_mirrored', 'tpu')
            devices: List of device names for MirroredStrategy (e.g., ['/gpu:0', '/gpu:1'])
            cross_device_ops: Communication backend for multi-GPU ('nccl', 'hierarchical_copy', 'reduction_to_one_device')
            tpu_address: TPU cluster address for TPUStrategy

        Returns:
            Configured tf.distribute.Strategy instance

        Raises:
            ValueError: If strategy_name is not valid
        """
        strategy_name = strategy_name.lower()

        if strategy_name not in DistributionStrategyFactory.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy_name}'. "
                f"Must be one of: {', '.join(sorted(DistributionStrategyFactory.VALID_STRATEGIES))}"
            )

        if strategy_name == 'none':
            return DistributionStrategyFactory._create_default_strategy()

        elif strategy_name == 'mirrored':
            return DistributionStrategyFactory._create_mirrored_strategy(
                devices=devices,
                cross_device_ops=cross_device_ops,
            )

        elif strategy_name == 'multi_worker_mirrored':
            return DistributionStrategyFactory._create_multi_worker_strategy(
                cross_device_ops=cross_device_ops,
            )

        elif strategy_name == 'tpu':
            return DistributionStrategyFactory._create_tpu_strategy(
                tpu_address=tpu_address,
            )

        # Should never reach here
        raise ValueError(f"Unknown strategy: {strategy_name}")

    @staticmethod
    def _create_default_strategy() -> tf.distribute.Strategy:
        """Create default (no-op) strategy for single device training."""
        return tf.distribute.get_strategy()

    @staticmethod
    def _create_mirrored_strategy(
        devices: Optional[List[str]] = None,
        cross_device_ops: str = 'nccl',
    ) -> tf.distribute.MirroredStrategy:
        """
        Create MirroredStrategy for multi-GPU training on a single node.

        Args:
            devices: List of GPU device strings (e.g., ['/gpu:0', '/gpu:1'])
                     If None, uses all available GPUs
            cross_device_ops: Communication backend ('nccl', 'hierarchical_copy', 'reduction_to_one_device')

        Returns:
            Configured MirroredStrategy
        """
        # Select cross-device communication ops
        if cross_device_ops == 'nccl':
            comm_ops = tf.distribute.NcclAllReduce()
        elif cross_device_ops == 'hierarchical_copy':
            comm_ops = tf.distribute.HierarchicalCopyAllReduce()
        elif cross_device_ops == 'reduction_to_one_device':
            comm_ops = tf.distribute.ReductionToOneDevice()
        else:
            # Default to NCCL for best GPU performance
            comm_ops = tf.distribute.NcclAllReduce()

        strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=comm_ops,
        )

        print(f"MirroredStrategy created with {strategy.num_replicas_in_sync} replicas")
        if devices:
            print(f"  Devices: {devices}")
        else:
            print(f"  Using all available GPUs")
        print(f"  Cross-device ops: {cross_device_ops}")

        return strategy

    @staticmethod
    def _create_multi_worker_strategy(
        cross_device_ops: str = 'nccl',
    ) -> tf.distribute.MultiWorkerMirroredStrategy:
        """
        Create MultiWorkerMirroredStrategy for multi-node distributed training.

        Requires TF_CONFIG environment variable to be set with cluster configuration.
        Use setup_tf_config() to generate TF_CONFIG from SLURM environment.

        Args:
            cross_device_ops: Communication backend

        Returns:
            Configured MultiWorkerMirroredStrategy
        """
        # Check for TF_CONFIG
        tf_config = os.environ.get('TF_CONFIG')
        if tf_config is None:
            raise EnvironmentError(
                "TF_CONFIG environment variable not set. "
                "Use setup_tf_config() or set manually for multi-worker training."
            )

        # Select communication options
        if cross_device_ops == 'nccl':
            comm_options = tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
            )
        else:
            comm_options = tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
            )

        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=comm_options,
        )

        # Parse TF_CONFIG for logging
        config = json.loads(tf_config)
        task = config.get('task', {})

        print(f"MultiWorkerMirroredStrategy created with {strategy.num_replicas_in_sync} replicas")
        print(f"  Task type: {task.get('type', 'unknown')}")
        print(f"  Task index: {task.get('index', 'unknown')}")
        print(f"  Communication: {cross_device_ops}")

        return strategy

    @staticmethod
    def _create_tpu_strategy(
        tpu_address: Optional[str] = None,
    ) -> tf.distribute.TPUStrategy:
        """
        Create TPUStrategy for TPU training.

        Args:
            tpu_address: TPU cluster address (e.g., 'grpc://10.0.0.1:8470')
                        If None, attempts to auto-detect

        Returns:
            Configured TPUStrategy
        """
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.TPUStrategy(resolver)

        print(f"TPUStrategy created with {strategy.num_replicas_in_sync} replicas")
        if tpu_address:
            print(f"  TPU address: {tpu_address}")

        return strategy


def setup_tf_config(
    worker_hosts: List[str],
    task_type: str = 'worker',
    task_index: int = 0,
    port: int = 12345,
) -> str:
    """
    Generate and set TF_CONFIG for MultiWorkerMirroredStrategy.

    Args:
        worker_hosts: List of worker hostnames (e.g., ['node1', 'node2'])
        task_type: Task type ('worker', 'chief', 'ps', 'evaluator')
        task_index: Index of this task among tasks of same type
        port: Port for worker communication

    Returns:
        TF_CONFIG JSON string (also sets environment variable)
    """
    # Build worker addresses
    worker_addresses = [f"{host}:{port}" for host in worker_hosts]

    tf_config = {
        'cluster': {
            'worker': worker_addresses,
        },
        'task': {
            'type': task_type,
            'index': task_index,
        },
    }

    tf_config_json = json.dumps(tf_config)
    os.environ['TF_CONFIG'] = tf_config_json

    print(f"TF_CONFIG set for {task_type} {task_index}")
    print(f"  Workers: {worker_addresses}")

    return tf_config_json


def setup_tf_config_from_slurm() -> str:
    """
    Generate TF_CONFIG from SLURM environment variables.

    Uses SLURM_JOB_NODELIST and SLURM_NODEID to configure workers.
    Requires 'scontrol' to expand node list.

    Returns:
        TF_CONFIG JSON string

    Raises:
        EnvironmentError: If required SLURM variables are not set
    """
    import subprocess

    # Get SLURM environment variables
    nodelist = os.environ.get('SLURM_JOB_NODELIST')
    node_id = os.environ.get('SLURM_NODEID')

    if nodelist is None:
        raise EnvironmentError("SLURM_JOB_NODELIST not set")
    if node_id is None:
        raise EnvironmentError("SLURM_NODEID not set")

    # Expand nodelist using scontrol
    try:
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        worker_hosts = result.stdout.strip().split('\n')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise EnvironmentError(f"Failed to expand SLURM nodelist: {e}")

    task_index = int(node_id)

    return setup_tf_config(
        worker_hosts=worker_hosts,
        task_type='worker',
        task_index=task_index,
        port=12345,
    )


def get_strategy_info(strategy: tf.distribute.Strategy) -> Dict[str, Any]:
    """
    Get information about a distribution strategy.

    Args:
        strategy: TensorFlow distribution strategy

    Returns:
        Dictionary with strategy information
    """
    info = {
        'type': type(strategy).__name__,
        'num_replicas_in_sync': strategy.num_replicas_in_sync,
    }

    # Add strategy-specific info
    if isinstance(strategy, tf.distribute.MirroredStrategy):
        info['devices'] = [d for d in strategy.extended.worker_devices]
    elif isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
        tf_config = os.environ.get('TF_CONFIG')
        if tf_config:
            config = json.loads(tf_config)
            info['cluster'] = config.get('cluster', {})
            info['task'] = config.get('task', {})

    return info


def calculate_global_batch_size(
    batch_size_per_replica: int,
    strategy: tf.distribute.Strategy,
) -> int:
    """
    Calculate global batch size from per-replica batch size.

    Args:
        batch_size_per_replica: Batch size for each replica
        strategy: Distribution strategy

    Returns:
        Global batch size (per_replica * num_replicas)
    """
    return batch_size_per_replica * strategy.num_replicas_in_sync
