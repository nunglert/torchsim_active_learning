"""Trajectory processors for ``remote_handler``'s processing layer.

Each is a single module (arg schema + cluster-side ``run`` + host-side
:class:`~remote_handler.processing.base.Processor` spec + a ``register`` call),
mirroring ``remote_handler``'s built-in ``processors/``. The stock shard driver
imports the processor's module on the cluster before resolving steps, so
``torchsim_AL`` must be installed in the cluster env.
"""

from torchsim_AL.processors.mace_eval import MaceEvalArgs, MaceEvalProcessor

__all__ = ["MaceEvalArgs", "MaceEvalProcessor"]
