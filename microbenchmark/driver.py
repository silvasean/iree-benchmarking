import dataclasses
import subprocess
import tempfile
from io import BytesIO
import os
import multiprocessing
import abc
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch_mlir
from torch_mlir import TensorPlaceholder

import iree.compiler as ireec

from . import executor


def _get_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keep-trace", type=str,
                        help="Path to keep the trace file. If not specified, the trace file will be a temporary file.")
    parser.add_argument("-c", "--only-check-correctness", default=False, action="store_true",
                        help="Only check numerical correctness of the code, rather than doing any actual benchmarking.")
    return parser


class BenchmarkSequence(abc.ABC):
    @abc.abstractmethod
    def invocations(self) -> Iterable[executor.Invocation]:
        pass

    @abc.abstractmethod
    def process_trace(self, df: pd.DataFrame):
        pass


@dataclasses.dataclass
class MatmulProblem:
    mkn: Tuple[int, int, int]
    mkn_dynamic: Tuple[bool, bool, bool]

    def as_identifier(self):
        m, k, n = self.mkn
        short_dynamic_str = "x".join("d" if d else "s" for d in self.mkn_dynamic)
        return f"mkn_{m}x{k}x{n}_mkn_dynamic_{short_dynamic_str}"


class MatmulSweep(BenchmarkSequence):
    def __init__(self, problems: List[MatmulProblem]):
        self.problems = problems

    def _compile_for_problem(self, problem: MatmulProblem):
        m, k, n = problem.mkn
        m_dynamic, k_dynamic, n_dynamic = problem.mkn_dynamic

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
        linalg_module_asm = linalg_module.operation.get_asm(
            enable_debug_info=True)
        return ireec.compile_str(
            linalg_module_asm,
            target_backends=["llvm-cpu"],
            extra_args=[
                "--mlir-disable-threading",
                # To make trace assembly clickable in Tracy.
                "--iree-llvm-link-embedded=false",
            ],
        )
    
    def invocations(self):
        for problem in self.problems:
            vmfb = self._compile_for_problem(problem)
            m, k, n = problem.mkn
            lhs = np.random.rand(m, k).astype(np.float32)
            rhs = np.random.rand(k, n).astype(np.float32)
            yield executor.Invocation(
                name=f"matmul_sweep_{problem.as_identifier()}",
                vmfb=vmfb,
                function_name="forward",
                inputs=[executor.InvocationInput(lhs, cold=False),
                        executor.InvocationInput(rhs, cold=False)],
                golden_result=np.matmul(lhs, rhs),
            )
    
    # TODO: Find a way to bracket the IR with trace zones so that we can
    # do the analysis post-hoc rather than needing to sketchily infer the
    # regions based on dispatch region names and the number of benchmark
    # iterations per Invocation.
    def process_trace(self, df: pd.DataFrame, benchmark_iters: List[int]):
        print(df)
        print(benchmark_iters)


def main():
    args = _get_argparse().parse_args()
    iree_tracy_bin_dir = os.environ["IREE_TRACY_BIN_DIR"]
    iree_tracy_capture = os.path.join(iree_tracy_bin_dir, "iree-tracy-capture")
    iree_tracy_csvexport = os.path.join(
        iree_tracy_bin_dir, "iree-tracy-csvexport")

    print(f"Parent pid is {os.getpid()}")
    # Ensure a clean environment for the child process.
    multiprocessing.set_start_method("spawn")
    parent_conn, child_conn = multiprocessing.Pipe()

    # Start executor process.
    p = multiprocessing.Process(
        target=executor.executor_process_entry_fn, args=(child_conn,))
    p.start()
    if args.only_check_correctness:
        parent_conn.send("only check correctness")
    else:
        parent_conn.send("do benchmarking")

    sweep = MatmulSweep([
        MatmulProblem(mkn=(512, 512, 512),
                      mkn_dynamic=(False, False, False)),
        MatmulProblem(mkn=(1024, 1024, 1024),
                      mkn_dynamic=(False, False, False)),
    ])
    benchmark_iters = []
    for invocation in sweep.invocations():
        parent_conn.send(invocation)
        if not args.only_check_correctness:
            benchmark_iters.append(parent_conn.recv())
    
    parent_conn.send("done sending invocations")

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
        subprocess.check_call(
            [iree_tracy_capture, "-f", "-o", trace_file_name])

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
    if not args.only_check_correctness:
        sweep.process_trace(df, benchmark_iters)
    print("DONE")


if __name__ == "__main__":
    main()
