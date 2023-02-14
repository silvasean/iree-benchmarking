import time
import os

import argparse
import numpy as np

import torch
import transformers

import torch_mlir
import iree.compiler as ireec
import iree.runtime as ireert


def benchmark_ns(
    f,
    run_duration_sec=5,
    max_samples=50,
):
    # Do a warmup run.
    f()

    overall_benchmarking_start_sec = time.monotonic()

    ns_elapsed_per_run = []
    while len(ns_elapsed_per_run) < max_samples and \
            time.monotonic() - overall_benchmarking_start_sec < run_duration_sec:
        ns_begin = time.perf_counter_ns()
        f()
        ns_end = time.perf_counter_ns()
        ns_elapsed_per_run.append(ns_end - ns_begin)

    assert len(ns_elapsed_per_run) > 0
    return ns_elapsed_per_run


def _suppress_warnings():
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the HuggingFace model to use.",
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=3,
        help="Number of sentences to use.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=17,
        help="Sequence length to use.",
    )
    parser.add_argument(
        "--elide_large_elements",
        default=True,
        action="store_false",
        help="Whether to elide large elements in the asm dump.",
    )
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action="store_true",
        help="Whether to print verbose output.",
    )
    return parser


def main(args):
    model_name = args.model_name
    num_sentences = args.num_sentences
    sequence_length = args.sequence_length
    FIXED_SENTENCES = ["Hello world!", "Goodbye world!", "Interesting worle!"]
    # Replicate the sentences enough times to get the desired number of sentences.
    sentences = [FIXED_SENTENCES[i %
                                 len(FIXED_SENTENCES)] for i in range(num_sentences)]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        torchscript=True,
    )
    model.eval()
    model.cuda()
    # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
    )["input_ids"].cuda()

    if args.verbose:
        print(f"tokens: {tokens}")
        with torch.no_grad():
            print(f"model output: {model(tokens)}")
    # Embedded inputs
    # Shape [num_sentences, sequence_length, EMBEDDING_SIZE]
    with torch.no_grad():
        attention_layer_input = model.bert.embeddings(tokens)
    assert attention_layer_input.shape == (
        num_sentences, sequence_length, 768), f"attention_layer_input.shape: {attention_layer_input.shape}"

    if args.verbose:
        print(f"attention_layer_input.shape: {attention_layer_input.shape}")

    attention_layer = model.bert.encoder.layer[0].attention.self

    # Attended inputs
    # The output of the attention layer has the same shape as the input.
    # Shape [num_sentences, sequence_length, EMBEDDING_SIZE]
    with torch.no_grad():
        attention_layer_output = attention_layer(attention_layer_input)[0]
    if args.verbose:
        print(f"attention_layer_output.shape: {attention_layer_output.shape}")

    class UnwrapSingleElementTupleLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden):
            return self.layer(hidden)[0]

    attention_module = UnwrapSingleElementTupleLayer(attention_layer)
    linalg_mlir = torch_mlir.compile(
        attention_module,
        [attention_layer_input],
        output_type="linalg-on-tensors",
        use_tracing=True
    )
    kwargs = {}
    if args.elide_large_elements:
        kwargs["large_elements_limit"] = 10
    with open("/tmp/attention_layer.mlir", "w") as f:
        f.write(linalg_mlir.operation.get_asm(**kwargs))

    vmfb = ireec.compile_str(
        linalg_mlir.operation.get_asm(),
        target_backends=["cuda"],
        extra_args=[
            "--iree-hal-cuda-llvm-target-arch=sm_80",
            "--iree-flow-dump-dispatch-graph",
            "--iree-flow-dump-dispatch-graph-output-file=/tmp/attention_layer.dot",
        ],
    )
    config = ireert.Config(driver_name="cuda")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)

    iree_module = ctx.modules.module

    iree_input = ireert.asdevicearray(
        iree_module._context.config.device, attention_layer_input.cpu())

    iree_output = torch.from_numpy(
        np.asarray(iree_module["forward"](iree_input)))
    print(f"iree_output: {iree_output.shape}")
    assert torch.allclose(iree_output, attention_layer_output.cpu(), atol=1e-2,
                          rtol=1e-2), "IREE results are not numerically correct"

    torch.set_float32_matmul_precision("high")  # Enable TF32

    print("iree")
    print(benchmark_ns(lambda: iree_module["forward"](
        iree_input), max_samples=10))
    print("eager")

    def eager():
        with torch.no_grad():
            attention_module(attention_layer_input)
    print(benchmark_ns(eager, max_samples=10))

    inductor_attention_module = torch.compile(
        attention_module, backend="inductor")
    print("inductor")

    def inductor():
        with torch.no_grad():
            inductor_attention_module(attention_layer_input)
    print(benchmark_ns(inductor, max_samples=10))

    # TODO: Share the weights and make sure that the results are the same.
    mha_layer = torch.nn.MultiheadAttention(
        embed_dim=768, num_heads=12, batch_first=True)
    mha_layer.eval()
    mha_layer.cuda()
    mha_layer(attention_layer_input,
              attention_layer_input, attention_layer_input)

    print("fused_mha")

    def fused_mha():
        with torch.no_grad():
            mha_layer(attention_layer_input,
                      attention_layer_input, attention_layer_input)
    print(benchmark_ns(fused_mha, max_samples=10))


if __name__ == "__main__":
    _suppress_warnings()
    parser = _get_argparse()
    args = parser.parse_args()
    main(args)
