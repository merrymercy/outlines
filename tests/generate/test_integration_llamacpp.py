import pytest

import outlines.generate as generate
import outlines.models as models
import outlines.samplers as samplers

TEST_MODEL = "./llama-test-model/TinyMistral-248M-v2-Instruct.Q4_K_M.gguf"


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )


@pytest.mark.parametrize(
    "generator_type,params", ((generate.text, []), (generate.regex, ("[0-9]",)))
)
def test_llamacpp_generation_api(model, generator_type, params):
    generator = generator_type(model, *params)

    res = generator("test")
    assert isinstance(res, str)

    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    res = generator("test", stop_at=".")
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res1 = generator("test", seed=1, max_tokens=10)
    res2 = generator("test", seed=1, max_tokens=10)
    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert res1 == res2


@pytest.mark.xfail(reason="Batch inference not available in `llama-cpp-python`.")
def test_llamacpp_batch_inference(model):
    generator = generate.text(model)
    res = generator(["test", "test1"])
    assert len(res) == 2


def test_llamacpp_sampling_params(model):
    generator = generate.text(model)

    params = {
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0,
    }
    res = generator("test", seed=1, max_tokens=10, **params)
    assert isinstance(res, str)


def test_llamacpp_greedy_sampling(model):
    sampler = samplers.greedy()
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)


def test_llamacpp_multinomial_sampling(model):
    sampler = samplers.multinomial()
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, temperature=1.0)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, top_k=1)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, top_p=0.5)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)


@pytest.mark.xfail(
    reason="The `llama-cpp-python` library's high-level interface does not allow to take several samples."
)
def test_llamacpp_several_samples(model):
    sampler = samplers.multinomial(3)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert len(res) == 3
    assert isinstance(res[0], str)
    assert isinstance(res[1], str)


@pytest.mark.xfail(
    reason="The llama-cpp-python library's high-level interface does not support beam search."
)
def test_llamacpp_beam_search(model):
    sampler = samplers.beam_search(1)
    generator = generate.text(model, sampler)
    res1 = generator("test")
    sampler = samplers.greedy()
    generator = generate.text(model, sampler)
    res2 = generator("test")
    assert res1 == res2

    sampler = samplers.beam_search(2)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert len(res) == 2
    assert res[0] != res[1]
