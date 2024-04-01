# Llama.cpp

!!! Installation

    You need to install the `llama-cpp-python` library to use the llama.cpp integration. See [this section](#install-with-different-backends) to see how to install `llama-cpp-python` to use with CuBLAS, Metal, etc.

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library][llamacpp]. Llamacpp allows to run quantized models on machines with limited compute.

## Load the model

You can initialize the model by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):

```python
from outlines import models

model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
```

This will download the model files to the hub cache folder and load the weights in memory.

You can also initialize the model by passing the path to the weights on your machine. Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from outlines import models
from llama_cpp import Llama

llm = Llama("./phi-2.Q4_K_M.gguf")
model = models.llamacpp(llm)
```

If you need more control, you can pass the same keyword arguments to the model as you would pass in the [llama-ccp-library][llamacpp]:

```python
from outlines import models

model = models.llamacpp(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf"
    n_gpu_layers=-1,  # to use GPU acceleration
)
```

**Main parameters:**

| Parameters | Type | Description | Default |
|------------|------|-------------|---------|
| `n_gpu_layers`| `int` | Number of layers to offload to GPU. If -1, all layers are offloaded | `0` | 
| `split_mode` | `int` | How to split the model across GPUs. `1` for layer-wise split, `2` for row-wise split | `1` | 
| `main_gpu` | `int` | Main GPU | `0` |
| `tensor_split` | `Optional[List[float]]` | How split tensors should be distributed accross GPUs. If `None` the model is not split. | `None` |
| `n_ctx` | `int` | Text context. Inference from the model if set to `0` | `512` |
| `n_threads` | `Optional[int]` | Number of threads to use for generation. All available threads if set to `None`.| `None` | 
| `verbose` | `bool` | Print verbose outputs to `stderr` | `True` |

See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for the full list of parameters.

### Load model on GPU


```python
from llama_cpp import Llama
from outlines import models

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)
model = models.LlamaCpp(llm)
```

### Load LoRa adapters

## Generate text

**Extra keyword arguments:**


## Install with different backends


[llamacpp]: https://github.com/abetlen/llama-cpp-python
