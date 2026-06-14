"""Reference client for the gpu-broker HTTP API.

Public surface:
- GpuBrokerClient  — context-manager-shaped acquire/release + status + admin
- BrokerHandle     — returned from `with broker.acquire(role)`; carries endpoint URL
- InfeasibleError  — raised when the broker returns 503 (insufficient VRAM, no eviction plan)
- BrokerError      — base class for all broker-side failures

Typical use:

    from client.gpu_broker_client import GpuBrokerClient

    broker = GpuBrokerClient(client_id="brain-raglite-mcp")
    with broker.acquire("rerank") as h:
        rerank_resp = requests.post(f"{h.endpoint}/rerank", json={...}).json()
"""
from .gpu_broker_client import (
    BrokerError,
    BrokerHandle,
    GpuBrokerClient,
    InfeasibleError,
    WorkerStatus,
)

__all__ = [
    "BrokerError",
    "BrokerHandle",
    "GpuBrokerClient",
    "InfeasibleError",
    "WorkerStatus",
]
