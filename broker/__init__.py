"""gpu-broker — VRAM-aware Docker orchestrator for single-GPU model serving.

Phase 3 lifecycle owner. Reads config/roles.yaml; manages worker containers
via the Docker SDK; exposes acquire/release/status/pin/unpin over HTTP.

See README.md for architecture diagram and API contract.
"""

__version__ = "0.1.0"
