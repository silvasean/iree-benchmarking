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
    parser.add_argument("-t", "--trace-file", type=str,
                        help="Path to keep the trace file. If not specified, the trace file will be a temporary file.")
    parser.add_argument("-d", "--dataframe-csv-file", type=str,
                        help="If specified, path to write the raw tracy pd.DataFrame file as CSV.")
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


@dataclasses.dataclass(frozen=True)
class MatmulProblem:
    mkn: Tuple[int, int, int]
    mkn_dynamic: Tuple[bool, bool, bool]

    def as_identifier(self):
        m, k, n = self.mkn
        short_dynamic_str = "x".join("d" if d else "s" for d in self.mkn_dynamic)
        return f"mkn_{m}x{k}x{n}_mkn_dynamic_{short_dynamic_str}"

def _find_row_for_problem(grouped: pd.DataFrame, problem: MatmulProblem):
    for row in grouped.itertuples():
        if row.Index.endswith("{}x{}x{}".format(*problem.mkn)):
            return row.total_exec_time_ns, row.num_workgroups_dispatched
    raise RuntimeError(f"Could not find row for problem {problem}")


class MatmulSweep(BenchmarkSequence):
    def __init__(self, problems: List[MatmulProblem]):
        self.problems = problems
        seen_mkn_values = set()
        for problem in problems:
            # TODO: Due to inability to bracket invocations with trace zones,
            # we cannot reliably handle duplicate mkn values.
            assert problem.mkn not in seen_mkn_values, "Duplicate mkn values"
            seen_mkn_values.add(problem.mkn)

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
    # TODO: We could set phony source locations to do this.
    def process_trace(self, df: pd.DataFrame, benchmark_iters: List[int]):
        print(f"benchmark_iters = {benchmark_iters}")
        grouped = df.groupby("name").agg({"exec_time_ns": ["sum", "count"]})
        grouped.columns = ["total_exec_time_ns", "num_workgroups_dispatched"]
        gflops = {}
        for problem, count in zip(self.problems, benchmark_iters):
            total_exec_time_ns, num_workgroups_dispatched = \
                _find_row_for_problem(grouped, problem)
            # TODO: Bracket invocations with properly labeled zones to
            # make this actually robust.
            assert num_workgroups_dispatched % count == 0, "Irregular number of workgroups per benchmark iteration"
            ns_per_invocation = total_exec_time_ns / count
            m, k, n = problem.mkn
            gflops[problem] = (m * k * n) / ns_per_invocation
        for problem, gflops in gflops.items():
            print(problem, gflops)



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
    BASELINE_SIZES = [
        32, 64, 96,
        128, 192, 256, 384,
        #512, 1024,
        #1536, 2048,
        #3072, 4096,
    ]
    sweep = MatmulSweep([
        MatmulProblem(mkn=(s, s, s), mkn_dynamic=(False, False, False))
        for s in BASELINE_SIZES
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

    if args.trace_file:
        trace_file = open(args.trace_file, "wb")
        trace_file_name = args.trace_file
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
    if args.dataframe_csv_file:
        df.to_csv(args.dataframe_csv_file)
    if not args.only_check_correctness:
        sweep.process_trace(df, benchmark_iters)
    print("DONE")


if __name__ == "__main__":
    main()
