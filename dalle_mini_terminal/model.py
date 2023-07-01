#!/usr/bin/env python3

import os
import random
from functools import partial
import datetime

import numpy as np
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image
from vqgan_jax.modeling_flax_vqgan import VQModel

def main(prompt, dalle_dir, vqgan_dir, output_dir):
    prompts = [prompt]

    model, params = DalleBart.from_pretrained(dalle_dir, revision=None, dtype=jnp.float32, _do_init=False)
    vqgan, vqgan_params = VQModel.from_pretrained(vqgan_dir, revision=None, _do_init=False)
    processor = DalleBartProcessor.from_pretrained(dalle_dir, revision=None)

    jax.local_device_count()

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

    key, subkey = jax.random.split(key)
    encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey), params, None, None, None, 10.0)
    encoded_images = encoded_images.sequences[..., 1:]

    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))

    if output_dir != ".":
        os.chdir(output_dir)

    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        img.save(datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".jpg", "JPEG")

