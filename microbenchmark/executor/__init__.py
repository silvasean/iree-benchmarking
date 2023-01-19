from typing import Tuple, Optional

from dataclasses import dataclass

import numpy as np

# The low-level types that are used to communicate with the child process.
@dataclass
class InvocationInput:
    array: np.ndarray
    # Ensure that input is out of cache when used.
    cold: bool

@dataclass
class Invocation:
    name: str
    vmfb: bytes
    function_name: str
    inputs: Tuple[InvocationInput, ...]
    # If present, only correctness will be checked.
    golden_result: Optional[np.ndarray]

def executor_process_entry_fn(conn):
    # This must be imported here lazily to avoid importing iree.runtime
    # in the parent process due to TRACY_NO_EXIT.
    from ._executor import process_entry_fn
    return process_entry_fn(conn)
