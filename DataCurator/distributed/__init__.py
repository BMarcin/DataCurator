"""Distributed pipeline (wiki/Distributed.md): controller, monitor, worker.

Purely additive to the standalone runner — reuses the per-record execution
layer (``Stage.run()`` and the modifiers) while orchestration moves to NATS
JetStream (work queue, done events, job records) and S3 (chunk data).
"""
