# Microbenchmarking framework based on Tracy

This is a microbenchmarking framework for IREE based on IREE's tracy
instrumentation.

# How to use

```
# Ensure -DIREE_BUILD_TRACY=ON
export IREE_TRACY_BIN_DIR=$IREE_BUILD_DIR/tracy
# Ensure -DIREE_ENABLE_RUNTIME_TRACING=ON
export PYTHONPATH=$IREE_BUILD_DIR/runtime/bindings/python
# Ensure you are using a non-tracy-instrumented `iree.compiler`.
# Such as installed from the nightly pip package.

# Run the microbenchmark driver.
python -m microbenchmark.driver
# Keep the trace file and trace file CSV export locally for inspection.
python -m microbenchmark.driver -t micro.tracy -d micro.csv
```

# Architecture

All execution of microbenchmarks happens in a child process (the "executor")
to isolate the traced code.

The parent process feeds a set of `executor.Invocation`s to the executor
through a pipe, and captures a trace of the result, which is then analyzed.

# Key Challenges

One of the key difficulties of this project is to perform microbenchmarks:

1. Orchestrated from Python, and

2. With rich access to the performance data of the program.

This is necessary to allow the flow to perform rich experiments, data analysis,
and visualization. Previous benchmarking efforts have been assembled out of
command line tools that have provided limited ability to scale the effort's
level of sophistication.

# Design issues

There is currently no reliable way for us to correlate the trace with
our executed program. We rely on implementation details of IREE such as
dispatch region zone name conventions, and the fact that IREE dispatches the
same set of workgroups for a given workload. It would be great if there
was a way for us to annotate our IR with zone names around the regions we
want to measure, so then we could total dispatch time for dispatch zones
enclosed in a region whose name we can reliably identify.

TODO: We could perhaps do this by munging the source locations of the generated
IR and looking at the location column in the CSV dump.
