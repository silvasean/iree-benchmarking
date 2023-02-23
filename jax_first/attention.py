# Prevent JAX from using all GPU memory.
# Otherwise IREE will not be able to allocate memory (or worse, it will page).
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# https://github.com/google/jax/issues/1222
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Attention, GPT2Config
import iree.runtime as ireert
import iree.compiler as ireec
from jax import random
import jax.numpy as jnp
import jax
import argparse
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"


# Parser with num_sentences, sequence_length, n_embd, n_head.
# And useful help descriptions

def _get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sentences", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--preset", type=int, default=None, choices=range(8),
                        help="""
Choose a preset size for the model (default: 0). 0-7 are approximately the
GPT-3 sizes from Table 2.1 in https://arxiv.org/pdf/2005.14165.pdf 
This corresponds to an approximately exponential increase in number of
parameters from 125M (preset=0) to 175B (preset=7).
This sets --embedding_size and --num_heads.
Note: These models are usually sharded across multiple devices, so
this setting does not necessariliy correspond to the single-node workload.
""")
    return parser


def _set_mlir_module_name(mlir_module, name: str):
    from jaxlib.mlir.ir import StringAttr
    mlir_module.operation.attributes["sym_name"] = StringAttr.get(
        name, mlir_module.context)


def _runit(f, n=10):
    for _ in range(n):
        f()


PRESET_TABLE = {
    0: (768, 12),
    1: (1024, 16),
    2: (1536, 16),
    3: (2048, 16),
    4: (2560, 32),
    5: (4096, 32),
    6: (5120, 40),
    7: (12288, 96),
}


def main(args):
    if args.preset is not None:
        args.embedding_size, args.num_heads = PRESET_TABLE[args.preset]
    num_sentences = args.num_sentences
    sequence_length = args.sequence_length
    embedding_size = args.embedding_size
    num_heads = args.num_heads
    assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"
    print(f"num_sentences={num_sentences}")
    print(f"sequence_length={sequence_length}")
    print(f"embedding_size={embedding_size}")
    print(f"num_heads={num_heads}")

    config = GPT2Config()
    config.n_embd = embedding_size
    config.n_head = num_heads

    rng = random.PRNGKey(1)
    example_input = random.normal(
        rng, (num_sentences, sequence_length, config.n_embd))

    model = FlaxGPT2Attention(config=config)
    vars = model.init(rng, example_input)
    jitted = jax.jit(model.apply)

    mhlo_module = jitted.lower(vars, example_input).compiler_ir(dialect="mhlo")
    _set_mlir_module_name(mhlo_module, "the_module")

    linalg_module_bytes = ireec.compile_str(
        mhlo_module.operation.get_asm(enable_debug_info=False),
        target_backends=["cuda"],
        input_type="mhlo",
        extra_args=["--compile-to=input", "-mlir-print-debuginfo=false"])

    with open("/tmp/attention.mlir", "wb") as f:
        f.write(linalg_module_bytes)

    vmfb = ireec.compile_str(
        linalg_module_bytes,
        target_backends=["cuda"],
        extra_args=[
            "--iree-hal-cuda-llvm-target-arch=sm_80",
            "--iree-flow-dump-dispatch-graph",
            "--iree-flow-dump-dispatch-graph-output-file=/tmp/attention.dot",
            "--iree-flow-enable-aggressive-fusion",
            # Allow large buffers.
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
        ],
    )

    config = ireert.Config(driver_name="cuda")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    module = ctx.modules.the_module
    iree_main_fn = module["main"]

    flat_inputs, _structure = jax.tree_util.tree_flatten((vars, example_input))
    print("flat input shapes", jax.tree_map(jnp.shape, flat_inputs))
    device_inputs = [ireert.asdevicearray(
        config.device, x) for x in flat_inputs]

    iree_result = iree_main_fn(*device_inputs)
    xla_result, = jitted(vars, example_input)
    if not jnp.allclose(jnp.asarray(iree_result), xla_result, atol=1e-2, rtol=1e-2):
        raise ValueError("IREE results don't match XLA")
    print("IREE is numerically correct!")

    _runit(lambda: jitted(vars, example_input))
    _runit(lambda: iree_main_fn(*device_inputs))


if __name__ == "__main__":
    parser = _get_argparse()
    args = parser.parse_args()
    main(args)
