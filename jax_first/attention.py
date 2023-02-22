# Prevent JAX from using all GPU memory.
# Otherwise IREE will not be able to allocate memory (or worse, it will page).
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# https://github.com/google/jax/issues/1222
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

import jax
import jax.numpy as jnp
from jax import random

import iree.compiler as ireec
import iree.runtime as ireert

from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Attention, GPT2Config

config = GPT2Config()
config.n_embd = 768
config.n_head = 12

rng = random.PRNGKey(1)

NUM_SENTENCES = 128
SEQUENCE_LENGTH = 512


model = FlaxGPT2Attention(config=config)

example_input = random.normal(
    rng, (NUM_SENTENCES, SEQUENCE_LENGTH, config.n_embd))
vars = model.init(rng, example_input)

jitted = jax.jit(model.apply)

mhlo_module = jitted.lower(vars, example_input).compiler_ir(dialect="mhlo")


def set_mlir_module_name(mlir_module, name: str):
    from jaxlib.mlir.ir import StringAttr
    mlir_module.operation.attributes["sym_name"] = StringAttr.get(
        name, mhlo_module.context)


set_mlir_module_name(mhlo_module, "the_module")

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
device_inputs = [ireert.asdevicearray(config.device, x) for x in flat_inputs]

iree_result = iree_main_fn(*device_inputs)
xla_result, = jitted(vars, example_input)
if not jnp.allclose(jnp.asarray(iree_result), xla_result, atol=1e-2, rtol=1e-2):
    raise ValueError("IREE results don't match XLA")
print("IREE is numerically correct!")


def runit(f, n=10):
    for _ in range(n):
        f()


runit(lambda: jitted(vars, example_input))
runit(lambda: iree_main_fn(*device_inputs))
