#!/usr/bin/env python3

# constants
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
N_PREDICTIONS = 8
GEN_TOP_K = None
GEN_TOP_P = None
TEMPERATURE = None
COND_SCALE = 10.0


# stdlib imports
import random
from functools import partial
import datetime


# pypi imports
import numpy as np
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image


# repo imports
from vqgan_jax.modeling_flax_vqgan import VQModel


# functions
def load_dalle_mini(repo: str, version: str | None):
    """Load DALL-E mini"""
    return DalleBart.from_pretrained(repo, revision=version, dtype=jnp.float16, _do_init=False)

def load_vqgan(repo: str, version: str | None):
    """Load VQGAN"""
    return VQModel.from_pretrained(repo, revision=version, _do_init=False)

def load_processor(repo, version):
    """Load DALL-E mini processor"""
    return DalleBartProcessor.from_pretrained(repo, revision=version)


def main(prompt, artifacts_dir):
    prompts = [prompt]

    # check how many devices are available
    jax.local_device_count()

    print("Loading DALL-E Mini model...")
    model, params = load_dalle_mini(artifacts_dir, None)
    print("Loading VQGAN model...")
    vqgan, vqgan_params = load_vqgan(VQGAN_REPO, VQGAN_COMMIT_ID)
    print("Loading BART encoder...")
    processor = load_processor(artifacts_dir, None)

    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return model.generate(**tokenized_prompt, prng_key=key, params=params, top_k=top_k, top_p=top_p, temperature=temperature, condition_scale=condition_scale)

    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    for _ in range(max(N_PREDICTIONS // jax.device_count(), 1)):
        key, subkey = jax.random.split(key)
        print("Generating image(s)...")
        encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey), params, GEN_TOP_K, GEN_TOP_P, TEMPERATURE, COND_SCALE)
        encoded_images = encoded_images.sequences[..., 1:]

        print("Decoding image(s)...")
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))

        for decoded_img in decoded_images:
            print("Saving image...")
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            img.save(datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".jpg", "JPEG")

