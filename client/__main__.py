"""Module entry-point so `python -m client status` works cleanly.

The CLI logic lives in `gpu_broker_client._cli`; this file only routes argv.
Keeping it separate avoids the `runpy` double-import warning that occurs when
`__init__.py` re-exports symbols from the module being run as `__main__`.
"""
import sys

from .gpu_broker_client import _cli

if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
