from typing import Tuple

import math
import os
import random
import sys
import time

import numpy as np

# We have to import iree.runtime after TRACY_NO_EXIT is set.
assert "iree.runtime" not in sys.modules, "`iree.runtime` is already imported for some reason?"
os.environ["TRACY_NO_EXIT"] = "1"
import iree.runtime as ireert
assert "iree.runtime" in sys.modules, "Did code formatting move `import iree.runtime` to the top?"

from . import Invocation, InvocationInput


class ConcreteInputGenerator:
    """Generate program inputs for an invocation.
    
    This class manages the set of IREE DeviceArray's that are fed to
    invocations. In the simplest case this only requires transferring the
    arrays to the device. But in the case that some inputs are specified
    to be cold (that is, out of cache), then we need to replicate the cold
    data enough times that the working set is larger than any cache.
    """

    def __init__(self, inputs: Tuple[InvocationInput, ...], device):

        # A size assumed to be bigger than any cache in the system.
        byte_size_bigger_than_any_cache = 100 * 1024 * 1024

        cold_size_bytes = 0
        for input in inputs:
            if input.cold:
                cold_size_bytes += input.array.nbytes
        if cold_size_bytes == 0:
            num_replicas_needed = 1
        else:
            num_replicas_needed = math.ceil(
                byte_size_bigger_than_any_cache / cold_size_bytes)

        single_invocation_inputs = [
            ireert.asdevicearray(device, input.array) for input in inputs]
        # For any cold inputs, replicate them. This is easier to write in-place.
        inputs_for_invocations = [
            single_invocation_inputs] * num_replicas_needed
        for replica in range(num_replicas_needed):
            for input_num, input in enumerate(inputs):
                if input.cold:
                    # TODO: Deepcopy without a host transfer.
                    inputs_for_invocations[replica][input_num] = ireert.asdevicearray(
                        device, np.asarray(inputs_for_invocations[replica][input_num]))

        # Shuffle the invocations to defeat any inter-invocation correlation.
        self._inputs_for_invocations = random.sample(
            inputs_for_invocations, k=len(inputs_for_invocations))
        self.index = 0

    def get_next_inputs(self):
        """Returns the next set of inputs for an invocation."""
        inputs = self._inputs_for_invocations[self.index]
        self.index += 1
        if self.index == len(self._inputs_for_invocations):
            self.index = 0
        return inputs


def process_entry_fn(conn):
    print(f"Child pid is: {os.getpid()}")
    action = conn.recv()
    assert action in ("do benchmarking", "only check correctness")
    only_check_correctness = action == "only check correctness"

    while True:
        invocation: Invocation = conn.recv()
        if invocation == "done sending invocations":
            break

        print(f"Processing invocation: {invocation.name}")
        config = ireert.Config(driver_name="local-sync")
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.from_flatbuffer(
            ctx.instance, invocation.vmfb)
        ctx.add_vm_module(vm_module)
        iree_module = ctx.modules.module
        device = iree_module._context.config.device
        fn = iree_module[invocation.function_name]
        input_generator = ConcreteInputGenerator(invocation.inputs, device)

        if only_check_correctness:
            iree_result = fn(*input_generator.get_next_inputs())
            result = np.asarray(iree_result)
            if not np.allclose(result, invocation.golden_result):
                raise RuntimeError(
                    f"Result mismatch for invocation {invocation.name}")
        else:
            run_duration_max_sec = 5
            max_iterations = 50

            begin_sec = time.perf_counter()
            num_iterations = 0
            while time.perf_counter() - begin_sec < run_duration_max_sec and num_iterations < max_iterations:
                iree_result = fn(*input_generator.get_next_inputs())
                num_iterations += 1
            conn.send(num_iterations)

    conn.send("done executing child work")
