# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keras implementation of StableDiffusion.

Credits:

- Original implementation:
  https://github.com/CompVis/stable-diffusion
- Initial TF/Keras port:
  https://github.com/divamgupta/stable-diffusion-tensorflow

The current implementation is a rewrite of the initial TF/Keras port by
Divam Gupta.
"""

import math
import warnings
from pathlib import Path

import numpy as np
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras, ops, random
from keras_cv.src.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.src.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.src.models.stable_diffusion.decoder import Decoder
from keras_cv.src.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.src.models.stable_diffusion.text_encoder import TextEncoder
from utils.lib.utils import save_to_gif

from guidance.latent_guidance import get_guidance
from predict import get_predict_on_batch_fn
from utils.keras_utils import cumprod_to_orig

MAX_PROMPT_LENGTH = 77
TIMESTEP_EMBEDDING_DIM = 320


@keras_cv_export("keras_cv.models.StableDiffusion")
class StableDiffusion:
    """Keras implementation of Stable Diffusion.

    Note that the StableDiffusion API, as well as the APIs of the sub-components
    of StableDiffusion (e.g. ImageEncoder, DiffusionModel) should be considered
    unstable at this point. We do not guarantee backwards compatability for
    future changes to these APIs.

    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text
    description (called a "prompt").

    Arguments:
        img_height: int, height of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        img_width: int, width of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        jit_compile: bool, whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Defaults to
            False.

    Example:

    ```python
    from keras_cv.models import StableDiffusion
    from PIL import Image

    model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
    img = model.text_to_image(
        prompt="A beautiful horse running through a field",
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123,  # Set this to always get the same image from the same prompt
    )
    Image.fromarray(img[0]).save("horse.png")
    print("saved at horse.png")
    ```

    References:
    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/CompVis/stable-diffusion)
    """  # noqa: E501

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        # UNet requires multiples of 2**7 = 128
        _img_height = round(img_height / 128) * 128
        _img_width = round(img_width / 128) * 128

        if _img_height != img_height or _img_width != img_width:
            warnings.warn(
                f"Image dimensions must be multiples of 128. "
                f"Rounding to nearest multiple: ({_img_height}, {_img_width})."
            )
        self.img_height = _img_height
        self.img_width = _img_width
        self.image_shape = [self.img_height, self.img_width, 3]

        # lazy initialize the component models and the tokenizer
        self._image_encoder = None
        self._text_encoder = None
        self._diffusion_model = None
        self._decoder = None
        self._tokenizer = None

        self.jit_compile = jit_compile

        self.alphas_bar = _ALPHAS_CUMPROD
        self.alphas = cumprod_to_orig(self.alphas_bar)

        self.context = None
        self.timestep = None
        self.track_progress = None
        self.latent_diffusion = True  # necessary for compatibility with DDIM model
        self.image_range = (-1, 1)  # data range for images to operate on

        # this doesn't work with jit, but without it and jit also doesn't work
        # best to use the predict_on_batch function and turn off jit
        # which is automatically done in latent_guidance funcs when StableDiffusion model
        # is detected
        self.predict_on_batch = get_predict_on_batch_fn()
        # self.predict_on_batch = lambda model, inputs: model(inputs)

    def diffusion_model(self, inputs):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be
        modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self.predict_on_batch(self._diffusion_model, inputs)

    def image_encoder(self, inputs):
        """image_encoder returns the VAE Encoder with pretrained weights.

        Usage:
        ```python
        sd = keras_cv.models.StableDiffusion()
        my_image = np.ones((512, 512, 3))
        latent_representation = sd.image_encoder.predict(my_image)
        ```
        """
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder()
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self.predict_on_batch(self._image_encoder, inputs)

    def decoder(self, inputs):
        """decoder returns the diffusion image decoder model with pretrained
        weights. Can be overriden for tasks where the decoder needs to be
        modified.
        """
        if self._decoder is None:
            self._decoder = Decoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._decoder.compile(jit_compile=True)
        return self.predict_on_batch(self._decoder, inputs)

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def tokenizer(self):
        """tokenizer returns the tokenizer used for text inputs.
        Can be overriden for tasks like textual inversion where the tokenizer
        needs to be modified.
        """
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def text_to_image(
        self,
        prompt,
        batch_size=1,
        num_steps=50,
        seed=None,
    ):
        encoded_text = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            batch_size=batch_size,
            num_steps=num_steps,
            seed=seed,
            guidance="no-guidance",
        )

    def posterior_sampling(
        self,
        image,
        operator,
        diffusion_steps=50,
        guidance_method="dps",
        guidance_kwargs=None,
        verbose=True,
        initialization=None,  # "smart",
        batch_size=1,
        prompt="",
        seed=None,
    ):
        if prompt != "":
            print(f"Using prompt: {prompt}")
        encoded_text = self.encode_text(prompt)

        # either measurement or guidance "no-guidance"
        assert image is not None or guidance_method == "no-guidance", (
            "Either measurement or guidance 'no-guidance' should be passed to "
            "posterior_sample"
        )

        # normalize measurement to [-1, 1] for image encoder
        # and latent space guidance
        # give a bit more headroom for noise
        if image is not None:
            assert (
                ops.min(image) >= -10.0
            ), f"Measurement range: {ops.min(image)}, {ops.max(image)}, should be [-1, 1]"
            assert (
                ops.max(image) <= 10.0
            ), f"Measurement range: {ops.min(image)}, {ops.max(image)}, should be [-1, 1]"

        if initialization == "smart" and guidance_method != "no-guidance":
            # initialize diffusion noise with Z_T ~ Ɛ(A.T * y)
            # rather than sampling from Gaussian
            latent_measurement = self.image_encoder(
                ops.expand_dims(operator.transpose(image), axis=0)
            )
            latent_measurement = ops.convert_to_tensor(latent_measurement)
            diffusion_noise = self._diffuse_forward(
                latent_measurement,
                index=diffusion_steps - 1,
                num_steps=diffusion_steps,
                seed=seed,
            )
            seed = None
        else:
            # will be sampled from Gaussian given the seed
            diffusion_noise = None

        return self.generate_image(
            encoded_text,
            measurement=image,
            operator=operator,
            batch_size=batch_size,
            num_steps=diffusion_steps,
            diffusion_noise=diffusion_noise,
            seed=seed,
            guidance=guidance_method,
            guidance_kwargs=guidance_kwargs,
            verbose=verbose,
        )

    def active_sampling(
        self,
        image,
        initial_operator,
        update_operator_fn,
        num_samples_to_take,
        sampling_window,
        guidance_method="dps",
        initial_samples=None,
        initial_step=0,
        image_shape=None,
        diffusion_steps=None,
        guidance_kwargs=None,
        verbose=False,
        plot_callback=None,
        plotting_interval=50,
        initialization=None,  # "smart",
        prompt="",
        seed=None,
    ):
        # bit hacky definitely refactor
        batch_size = image_shape[0]

        if prompt != "":
            print(f"Using prompt: {prompt}")
        encoded_text = self.encode_text(prompt)

        # either measurement or guidance "no-guidance"
        assert image is not None or guidance_method == "no-guidance", (
            "Either measurement or guidance 'no-guidance' should be passed to "
            "posterior_sample"
        )

        # normalize measurement to [-1, 1] for image encoder
        # and latent space guidance
        # give a bit more headroom for noise
        if image is not None:
            assert (
                ops.min(image) >= -10.0
            ), f"Measurement range: {ops.min(image)}, {ops.max(image)}, should be [-1, 1]"
            assert (
                ops.max(image) <= 10.0
            ), f"Measurement range: {ops.min(image)}, {ops.max(image)}, should be [-1, 1]"

        operator = initial_operator
        start_sampling, stop_sampling = sampling_window
        assert start_sampling >= initial_step, (
            "sampling_window = (start, stop), where start should be >= initial_step, "
            f"got sampling_window = {sampling_window}"
        )
        assert stop_sampling <= diffusion_steps, (
            "sampling_window = (start, stop), where stop should be <= diffusion_steps, "
            f"got sampling_window = {sampling_window}"
        )
        sampling_interval = (stop_sampling - start_sampling) // num_samples_to_take
        sampling_interval = max(sampling_interval, 1)
        measurements = operator.forward(image)
        run_guidance = self.get_guidance_fn(
            guidance_method,
            operator,
        )

        if initialization == "smart" and guidance_method != "no-guidance":
            # initialize diffusion noise with Z_T ~ Ɛ(A.T * y)
            # rather than sampling from Gaussian
            latent_measurement = self.image_encoder(
                ops.expand_dims(operator.transpose(image), axis=0)
            )
            latent_measurement = ops.convert_to_tensor(latent_measurement)
            initial_noise = self._diffuse_forward(
                latent_measurement,
                index=diffusion_steps - 1,
                num_steps=diffusion_steps,
                seed=seed,
            )
            seed = None
        else:
            # will be sampled from Gaussian given the seed
            initial_noise = None

        if initial_noise is not None and seed is not None:
            raise ValueError(
                "`initial_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        if guidance_kwargs is None:
            guidance_kwargs = {}

        context = self._expand_tensor(encoded_text, batch_size)
        # context (i.e. prompt) is fixed here and usually set to empty for inverse process
        self.context = context

        if initial_noise is not None:
            initial_noise = ops.squeeze(initial_noise)
            if len(ops.shape(initial_noise)) == 3:
                initial_noise = ops.repeat(
                    ops.expand_dims(initial_noise, axis=0), batch_size, axis=0
                )
            latent = initial_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        assert diffusion_steps <= 1000, "diffusion_steps should be <= 1000"

        timesteps = np.arange(1, 1000, 1000 // diffusion_steps)
        alphas_bar, alphas_bar_prev = self._get_alphas_bar_timestep(timesteps)

        progbar = keras.utils.Progbar(len(timesteps), verbose=verbose)
        iteration = 0
        error = None

        # grad_compute_guidance_error = get_guidance(guidance)(self)
        run_guidance = self.get_guidance_fn(guidance_method, operator)
        for index, timestep in list(enumerate(timesteps))[::-1]:
            self.timestep = timestep
            a_bar_t, a_bar_prev = alphas_bar[index], alphas_bar_prev[index]
            signal_rates = ops.sqrt(a_bar_t)
            noise_rates = ops.sqrt(1.0 - a_bar_t)
            next_signal_rates = ops.sqrt(a_bar_prev)
            next_noise_rates = ops.sqrt(1.0 - a_bar_prev)

            latent, error, pred_images = run_guidance(
                latent,
                measurement=image,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                next_signal_rates=next_signal_rates,
                next_noise_rates=next_noise_rates,
                **guidance_kwargs,
            )

            pred_images = self.decoder(pred_images)

            if (
                iteration >= start_sampling
                and iteration < stop_sampling
                and iteration % sampling_interval == 0
            ):

                operator = update_operator_fn(
                    pred_images,
                )
                measurements = operator.forward(image)

                if verbose and plot_callback is not None:
                    plot_callback.add_to_buffer(
                        iteration, measurements, pred_images, latent
                    )

            elif iteration % plotting_interval == 0:
                if verbose and plot_callback is not None:
                    plot_callback.add_to_buffer(
                        iteration, measurements, pred_images, latent
                    )

            iteration += 1
            if error is not None:
                progbar.update(iteration, [("measurement_error", error)])
            else:
                progbar.update(iteration)

        return pred_images, measurements, operator

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.

        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding prior to image generation, e.g. for walking
        between two prompts.

        Args:
            prompt: a string to encode, must be 77 tokens or shorter.

        Example:

        ```python
        from keras_cv.models import StableDiffusion

        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        encoded_text  = model.encode_text("Tacos at dawn")
        img = model.generate_image(encoded_text)
        ```
        """
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = ops.convert_to_tensor([phrase], dtype="int32")

        context = self.text_encoder.predict_on_batch(
            {"tokens": phrase, "positions": self._get_pos_ids()}
        )

        return context

    def generate_image(
        self,
        encoded_text,
        measurement=None,
        operator=None,
        batch_size=1,
        num_steps=50,
        diffusion_noise=None,
        seed=None,
        guidance="dps",
        guidance_kwargs=None,
        track_progress=True,
        animate_latent=False,
        verbose=True,
    ):
        """Generates an image based on encoded text.

        The encoding passed to this method should be derived from
        `StableDiffusion.encode_text`.

        Args:
            encoded_text: Tensor of shape (`batch_size`, 77, 768), or a Tensor
                of shape (77, 768). When the batch axis is omitted, the same
                encoded text will be used to produce every generated image.
            batch_size: int, number of images to generate, defaults to 1.
            negative_prompt: a string containing information to negatively guide
                the image generation (e.g. by removing or altering certain
                aspects of the generated image), defaults to None.
            measurement: Tensor of shape (`batch_size`, img_height, img_width, 3),
                and image range [-1, 1].
            num_steps: int, number of diffusion steps (controls image quality),
                defaults to 50.
            unconditional_guidance_scale: float, controlling how closely the
                image should adhere to the prompt. Larger values result in more
                closely adhering to the prompt, but will make the image noisier.
                Defaults to 7.5.
            diffusion_noise: Tensor of shape (`batch_size`, img_height // 8,
                img_width // 8, 4), or a Tensor of shape (img_height // 8,
                img_width // 8, 4). Optional custom noise to seed the diffusion
                process. When the batch axis is omitted, the same noise will be
                used to seed diffusion for every generated image.
            seed: integer which is used to seed the random generation of
                diffusion noise, only to be specified if `diffusion_noise` is
                None.
            guidance: string, the name of the guidance function to use. Defaults
                to "dps". See `guidance.py` module for available guidance.

        Example:

        ```python
        from keras_cv.models import StableDiffusion
        from keras_core import ops

        batch_size = 8
        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        e_tacos = model.encode_text("Tacos at dawn")
        e_watermelons = model.encode_text("Watermelons at dusk")

        e_interpolated = ops.linspace(e_tacos, e_watermelons, batch_size)
        images = model.generate_image(e_interpolated, batch_size=batch_size)
        ```
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        if guidance_kwargs is None:
            guidance_kwargs = {}

        context = self._expand_tensor(encoded_text, batch_size)
        # context (i.e. prompt) is fixed here and usually set to empty for inverse process
        self.context = context

        if diffusion_noise is not None:
            diffusion_noise = ops.squeeze(diffusion_noise)
            if len(ops.shape(diffusion_noise)) == 3:
                diffusion_noise = ops.repeat(
                    ops.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        if track_progress:
            self.track_progress = []
            print(
                "Warning, track_progress is turned on, which slows down "
                "the generation process (only at the end)."
            )
        # Iterative reverse diffusion stage
        assert num_steps <= 1000, "num_steps should be <= 1000"

        timesteps = np.arange(1, 1000, 1000 // num_steps)
        alphas_bar, alphas_bar_prev = self._get_alphas_bar_timestep(timesteps)

        progbar = keras.utils.Progbar(len(timesteps), verbose=verbose)
        iteration = 0
        error = None

        self.track_progress = []
        if num_steps > 50:
            track_progress_interval = num_steps // 50
        else:
            track_progress_interval = 1

        # grad_compute_guidance_error = get_guidance(guidance)(self)
        run_guidance = self.get_guidance_fn(guidance, operator)
        for index, timestep in list(enumerate(timesteps))[::-1]:
            self.timestep = timestep
            a_bar_t, a_bar_prev = alphas_bar[index], alphas_bar_prev[index]
            signal_rates = ops.sqrt(a_bar_t)
            noise_rates = ops.sqrt(1.0 - a_bar_t)
            next_signal_rates = ops.sqrt(a_bar_prev)
            next_noise_rates = ops.sqrt(1.0 - a_bar_prev)

            ## this is using dps_score
            # gradients, (error, (pred_z0, score_t)) = grad_compute_guidance_error(
            #     latent,
            #     timestep=timestep,
            #     a_bar_t=a_bar_t,
            #     operator=operator,
            #     measurement=measurement,
            # )
            # latent = self._compute_next_latent(score_t, pred_z0, a_bar_t, a_bar_prev)
            # latent = latent - gradients

            latent, error, pred_images = run_guidance(
                latent,
                measurement=measurement,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                next_signal_rates=next_signal_rates,
                next_noise_rates=next_noise_rates,
                **guidance_kwargs,
            )

            if index % track_progress_interval == 0:
                self.track_progress.append(pred_images)

            iteration += 1
            if error is not None:
                progbar.update(iteration, [("measurement_error", error)])
            else:
                progbar.update(iteration)

        if animate_latent:
            pred_images = ops.convert_to_numpy(pred_images)
            # Decoding stage
            if track_progress:
                if verbose:
                    print("Animating latent optimization process...")
                try:
                    self.anim_latent_optimization(track_progress)
                except:
                    print("Animation failed, continuing without it.")

        decoded = self.decoder(pred_images)

        return decoded

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        training=False,
    ):
        # predict noise component and calculate the image component using it
        batch_size = ops.shape(noisy_images)[0]
        t_emb = self._get_timestep_embedding(self.timestep, batch_size)
        inputs = {
            "latent": noisy_images,
            "timestep_embedding": t_emb,
            "context": self.context,
        }

        pred_noises = self.diffusion_model(inputs)

        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def get_guidance_fn(
        self,
        guidance_method,
        operator,
    ):
        guidance_fn = get_guidance(guidance_method)(self)

        def run_guidance(
            noisy_images,
            measurement,
            noise_rates,
            signal_rates,
            next_signal_rates,
            next_noise_rates,
            **guidance_kwargs,
        ):

            gradients, (measurement_error, (pred_noises, pred_images)) = guidance_fn(
                noisy_images,
                measurement=measurement,
                operator=operator,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                **guidance_kwargs,
            )
            gradients = ops.nan_to_num(gradients)
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            next_noisy_images = next_noisy_images - gradients
            return next_noisy_images, measurement_error, pred_images

        return run_guidance

    def _decode_latent_to_image(self, latent):
        decoded = self.decoder(latent)
        # convert from [-1, 1] to [0, 255]
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _compute_score_t(self, latent, timestep, a_bar_t):
        """compute score sθ (z_t, t) given current latent z_t"""
        batch_size = ops.shape(latent)[0]
        t_emb = self._get_timestep_embedding(timestep, batch_size)
        inputs = {
            "latent": latent,
            "timestep_embedding": t_emb,
            "context": self.context,
        }

        pred_noise = self.diffusion_model(inputs)
        # pred_noise = self.diffusion_model(input)
        score_t = -1 / ops.sqrt(1 - a_bar_t) * pred_noise
        return score_t

    def _noise_to_score(self, noise, a_bar_t):
        """compute score sθ (z_t, t) given current noise"""
        score_t = -1 / ops.sqrt(1 - a_bar_t) * noise
        return score_t

    def _compute_pred_z0(self, z_t, score_t, a_bar_t):
        """predict the final latent z_0 from the current latent z_t
        z_0_tweedie ← (z_t + (1 - alpha_t) sθ (z_t, t))/sqrt(a_t)
        """
        pred_z0 = (z_t + (1 - a_bar_t) * score_t) / ops.sqrt(a_bar_t)
        return pred_z0

    def _compute_next_latent(self, score_t, pred_z0, a_bar_t, a_bar_prev):
        next_latent = (
            -ops.sqrt(1 - a_bar_t) * score_t * ops.sqrt(1.0 - a_bar_prev)
            + ops.sqrt(a_bar_prev) * pred_z0
        )
        return next_latent

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch
        size."""
        text_embedding = ops.squeeze(text_embedding)
        if len(text_embedding.shape) == 2:
            text_embedding = ops.repeat(
                ops.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    def _get_timestep_embedding(self, timestep, batch_size, max_period=10000):
        half = TIMESTEP_EMBEDDING_DIM // 2
        range_ = ops.cast(ops.arange(0, half), "float32")
        freqs = ops.exp(-math.log(max_period) * range_ / half)
        args = ops.convert_to_tensor([timestep], dtype="float32") * freqs
        embedding = ops.concatenate([ops.cos(args), ops.sin(args)], 0)
        embedding = ops.reshape(embedding, [1, -1])
        return ops.repeat(embedding, batch_size, axis=0)

    def _get_alphas_timestep(self, timesteps):
        alphas = [self.alphas[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_alphas_bar_timestep(self, timesteps):
        alphas_bar = [self.alphas_bar[t] for t in timesteps]
        alphas_bar_prev = [1.0] + alphas_bar[:-1]

        return alphas_bar, alphas_bar_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        return random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4),
            seed=seed,
        )

    def _diffuse_forward(self, latent_clean, index, num_steps, seed=None):
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        alphas_bar, _ = self._get_alphas_bar_timestep(timesteps)

        a_bar_t = alphas_bar[index]

        batch_size = ops.shape(latent_clean)[0]
        eps = self._get_initial_diffusion_noise(batch_size, seed)
        # use special property of forward diffusion process to do p_t(z_t|z_0)
        # z_t = z_0 * sqrt(alpha_bar_t) + sqrt(1 - alpha_bar_t) * eps, where eps ~ N(0, 1)
        latent = ops.sqrt(a_bar_t) * latent_clean + ops.sqrt(1.0 - a_bar_t) * eps
        return latent

    @staticmethod
    def _get_pos_ids():
        return ops.expand_dims(ops.arange(MAX_PROMPT_LENGTH, dtype="int32"), 0)

    def anim_latent_optimization(
        self, latents, path: str = None, batch_size: int = 1, duration: float = 4
    ):
        """Animates the latent optimization process.

        Args:
            latents (list): list with latent Tensors of shape
                (`batch_size`, img_height // 8, img_width // 8, 4).
            path (str, optional): saving path for animation. Defaults to None.
                in that case animation is saved to "assets/animation.gif".
            batch_size (int, optional): batch size of decoding latents into images.
                Defaults to 1.
            duration (float, optional): duration of animation in seconds.
                Defaults to 4.
        """
        # stack list of tensors into one tensor of shape
        # (num_steps, batch_size, img_height // 8, img_width // 8, 4)
        latents = ops.stack(latents)

        num_steps = ops.shape(latents)[0]
        # reshape to (num_steps * batch_size, img_height // 8, img_width // 8, 4)
        latents = ops.reshape(latents, (-1, *ops.shape(latents)[2:]))

        # decode latent to image in batches
        images = []
        progbar = keras.utils.Progbar(np.ceil(len(latents) / batch_size))
        j = 0
        for i in range(0, len(latents), batch_size):
            latent = latents[i : i + batch_size]
            decoded = self._decode_latent_to_image(latent)
            images.append(decoded)
            j += 1
            progbar.update(j)

        # reshape back to (num_steps, batch_size, img_height, img_width, 3)
        images = np.concatenate(images)
        images = np.reshape(images, (num_steps, -1, *np.shape(images)[1:]))

        # repeat the last image in each batch to see the result in the end
        # add about 10% of total length
        images = np.concatenate(
            [images, np.repeat(images[-1:], int(num_steps * 0.1), axis=0)], axis=0
        )
        # length changed due to adding the last image repeatetly
        num_steps = len(images)

        # for now only save first image in batch
        images = images[:, 0]

        # Convert the list of frames to an animated GIF
        if path is None:
            path = "assets/animation.gif"
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        # make sure the animation is always 4 sec
        fps = num_steps // duration
        # apparently above 50 doesn't work
        fps = min(fps, 50)
        save_to_gif(images, str(path), fps=fps)

    def compile(self, jit_compile=True):
        self.jit_compile = jit_compile
        # just passing dummy data will compile models
        print("Compiling StableDiffusion...")
        self.image_encoder(
            ops.ones([1, self.img_height, self.img_width, 3], dtype="float32")
        )
        self.decoder(
            ops.ones([1, self.img_height // 8, self.img_width // 8, 4], dtype="float32")
        )
        self.diffusion_model(
            {
                "latent": ops.ones([1, self.img_height // 8, self.img_width // 8, 4]),
                "timestep_embedding": ops.ones(
                    [1, TIMESTEP_EMBEDDING_DIM], dtype="float32"
                ),
                "context": ops.ones([1, MAX_PROMPT_LENGTH, 768], dtype="float32"),
            }
        )

        self.text_encoder(
            {
                "tokens": ops.ones([1, MAX_PROMPT_LENGTH], dtype="int32"),
                "positions": ops.ones([1, MAX_PROMPT_LENGTH], dtype="int32"),
            }
        )
