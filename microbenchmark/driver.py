import subprocess
import tempfile


from io import BytesIO
import os
import multiprocessing

import numpy as np
import pandas as pd

import torch
import torch_mlir
from torch_mlir import TensorPlaceholder

from . import executor

M, K, N = 1024, 1024, 1024

def _get_vmfb():
    import iree.compiler as ireec
    m, k, n = M, K, N
    m_dynamic, k_dynamic, n_dynamic = False, False, False
    class MatmulModule(torch.nn.Module):
        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs)

    lhs_placeholder = TensorPlaceholder([
        -1 if m_dynamic else m,
        -1 if k_dynamic else k,
    ], dtype=torch.float32)
    rhs_placeholder = TensorPlaceholder([
        -1 if k_dynamic else k,
        -1 if n_dynamic else n,
    ], dtype=torch.float32)

    linalg_module = torch_mlir.compile(
        MatmulModule(),
        (lhs_placeholder, rhs_placeholder),
        output_type="linalg-on-tensors",
    )
    linalg_module_asm = linalg_module.operation.get_asm(enable_debug_info=True)
    return ireec.compile_str(linalg_module_asm,
        target_backends=["llvm-cpu"],
        extra_args=[
            "--mlir-disable-threading",
            # To make trace assembly clickable in Tracy.
            # TODO: This isn't working, but suppo
            "--iree-llvm-link-embedded=false",
        ],
    )


def _get_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keep-trace", type=str, help="Path to keep the trace file. If not specified, the trace file will be a temporary file.")
    parser.add_argument("--iree-tracy-bin-dir", type=str,
                        help="Path to the directory holding iree-tracy-capture and iree-tracy-csvexport tools. Usually `$IREE_BUILD_DIR/tracy`")
    return parser

def main():
    args = _get_argparse().parse_args()
    iree_tracy_bin_dir = os.environ["IREE_TRACY_BIN_DIR"]
    iree_tracy_capture = os.path.join(iree_tracy_bin_dir, "iree-tracy-capture")
    iree_tracy_csvexport = os.path.join(iree_tracy_bin_dir, "iree-tracy-csvexport")

    print(f"Parent pid is {os.getpid()}")
    # Ensure a clean environment for the child process.
    multiprocessing.set_start_method("spawn")
    vmfb = _get_vmfb()
    parent_conn, child_conn = multiprocessing.Pipe()

    # Start executor process.
    p = multiprocessing.Process(
        target=executor.executor_process_entry_fn, args=(child_conn,))
    p.start()


    lhs = np.random.rand(M, K).astype(np.float32)
    rhs = np.random.rand(K, N).astype(np.float32)
    invocation = executor.Invocation(
        name="simple_matmul_1024x1024x1024",
        vmfb=vmfb,
        function_name="forward",
        inputs=[executor.InvocationInput(lhs, cold=False),
                executor.InvocationInput(rhs, cold=False)],
        golden_result=np.matmul(lhs, rhs),
    )
    parent_conn.send(invocation)

    # End of executor process.
    wait_result = parent_conn.recv()
    assert wait_result == "done executing child work"

    if args.keep_trace:
        trace_file = open(args.keep_trace, "wb")
        trace_file_name = args.keep_trace
    else:
        trace_file = tempfile.NamedTemporaryFile(
            suffix=".tracy",
        )
        trace_file_name = trace_file.name
    with trace_file:
        subprocess.check_call([iree_tracy_capture, "-f", "-o", trace_file_name])

        # Joining needs to happen after we capture the trace because
        # TRACY_NO_EXIT will halt the child process until we do.
        p.join()

        tracy_csv = subprocess.check_output([
            iree_tracy_csvexport,
            "-u",
            "-f", "forward_dispatch",
            trace_file_name,
        ])

    df = pd.read_csv(BytesIO(tracy_csv))
    print(df)
    print("DONE")

if __name__ == "__main__":
    main()
