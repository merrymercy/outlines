import dataclasses
from typing import TYPE_CHECKING, List, Optional, TypedDict, Union

from typing_extensions import Unpack

from outlines.generate.api import GenerationParameters, SamplingParameters

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList


class LlamaCppParams(TypedDict, total=False):
    suffix: Optional[str]
    temperature: float
    top_p: float
    min_p: float
    typical_p: float
    seed: int
    max_tokens: int
    logits_processor: "LogitsProcessorList"
    stop: Optional[Union[str, List[str]]]
    frequence_penalty: float
    presence_penalty: float
    repeat_penalty: float
    top_k: int
    tfs_z: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float


class LlamaCpp:
    """Represents a model provided by the `llama-cpp-python` library.

    We wrap models from model providing libraries in order to give all of
    them the same interface in Outlines and allow users to easily switch
    between providers. This class wraps the `llama_cpp.Llama` class from the
    `llama-cpp-python` library.

    """

    def __init__(self, model: "Llama"):
        self.model = model

    def generate(
        self,
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ):
        """Generate text using `llama-cpp-python`.

        Arguments
        ---------
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        structure_logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        llama_cpp_params
            Keyword arguments that can be passed to
            `llama_cpp_python.Llama.__call__`.  The values in `llama_cpp_params`
            supersed the values of the parameters in `generation_parameters` and
            `sampling_parameters`.  See the `llama_cpp_python` documentation for
            a list of possible values: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__

        Returns
        -------
        The generated text.

        """
        from llama_cpp import LogitsProcessorList

        prompts, max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        # We update `llama_cpp_params` with the values the user passed to the
        # generator.
        if "stop" not in llama_cpp_params:
            llama_cpp_params["stop"] = stop_at
        if "seed" not in llama_cpp_params:
            llama_cpp_params["seed"] = seed
        if "max_tokens" not in llama_cpp_params:
            llama_cpp_params["max_tokens"] = max_tokens

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        # We update the `llama_cpp_params` with the sampling values that
        # were specified by the user via the `Sampler` class, unless they
        # are also specified in `llama_cpp_params`.
        if sampler == "beam_search":
            raise NotImplementedError(
                "The `llama_cpp_python` library does not support Beam Search."
            )
        if num_samples != 1:
            raise NotImplementedError(
                "The `llama_cpp_python` library does not allow to take several samples."
            )
        if "top_p" not in llama_cpp_params and top_p is not None:
            llama_cpp_params["top_p"] = top_p
        if "top_k" not in llama_cpp_params and top_k is not None:
            llama_cpp_params["top_k"] = top_k
        if "temperature" not in llama_cpp_params and temperature is not None:
            llama_cpp_params["temperature"] = temperature

        if structure_logits_processor is not None:
            if "logits_processor" in llama_cpp_params:
                llama_cpp_params["logits_processor"].append(structure_logits_processor)
            else:
                llama_cpp_params["logits_processor"] = LogitsProcessorList(
                    [structure_logits_processor]
                )

        completion = self.model(prompts, **llama_cpp_params)
        result = completion["choices"][0]["text"]

        self.model.reset()

        return result

    def stream(self):
        raise NotImplementedError


def llamacpp(
    repo_id: str, filename: Optional[str] = None, **llamacpp_model_params
) -> LlamaCpp:
    """Load a model from the `llama-cpp-python` library.

    We use the `Llama.from_pretrained` classmethod that downloads models
    directly from the HuggingFace hub, instead of asking users to specify
    a path to the downloaded model. One can still load a local model
    by initializing `llama_cpp.Llama` directly.

    Arguments
    ---------
    repo_id
        The name of the model repository.
    filename:
        A filename of glob pattern to match the model file in the repo.
    llama_cpp_model_params
        Llama-specific model parameters. See the `llama-cpp-python` documentation
        for the full list: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    """
    from llama_cpp import Llama

    model = Llama.from_pretrained(repo_id, filename, **llamacpp_model_params)

    return LlamaCpp(model)
