
// See requirements.txt
python -m pip install -f https://iree-org.github.io/iree/pip-release-links.html iree-compiler iree-runtime
python -m pip install -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
python -m pip install seaborn


// Whole-program GPU trace overview
nsys profile --gpu-metrics-device all python MatmulSweep.py
// Instruction-level granularity of single kernel
// Get this command line from the nsys UI if possible.
ncu -o foo2048 --kernel-name forward_dispatch_0_matmul_2048x2048x2048 --launch-skip 14 --launch-count 1 "python" MatmulSweep.py
