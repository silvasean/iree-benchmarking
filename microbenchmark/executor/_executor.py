import sys
import os

import numpy as np

# We have to import iree.runtime after TRACY_NO_EXIT is set.
assert "iree.runtime" not in sys.modules
os.environ["TRACY_NO_EXIT"] = "1"
import iree.runtime as ireert
assert "iree.runtime" in sys.modules

from . import Invocation, InvocationInput

def process_entry_fn(conn):
    print(f"Child pid is: {os.getpid()}")
    config = ireert.Config(driver_name="local-sync")
    ctx = ireert.SystemContext(config=config)
    invocation: Invocation = conn.recv()

    print(f"Processing invocation: {invocation.name}")
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, invocation.vmfb)
    ctx.add_vm_module(vm_module)
    iree_module = ctx.modules.module

    inputs = []
    for input in invocation.inputs:
        device_array = ireert.asdevicearray(iree_module._context.config.device, input.array)
        inputs.append(device_array)

    if invocation.golden_result is not None:
        iree_result = iree_module[invocation.function_name](*inputs)
        result = np.asarray(iree_result)
        if not np.allclose(result, invocation.golden_result):
            raise RuntimeError(f"Result mismatch for invocation {invocation.name}")

    conn.send("done executing child work")
