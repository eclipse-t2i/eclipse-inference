import gradio as gr
from PIL import Image

import torch

from torchvision import transforms
from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPFeatureExtractor,
)

import math
from typing import List
from PIL import Image, ImageChops
import numpy as np
import torch

from diffusers import UnCLIPPipeline

# from diffusers.utils.torch_utils import randn_tensor

from transformers import CLIPTokenizer

from src.priors.prior_transformer import (
    PriorTransformer,
)  # original huggingface prior transformer without time conditioning
from src.pipelines.pipeline_kandinsky_prior import KandinskyPriorPipeline

from diffusers import DiffusionPipeline


__DEVICE__ = "cpu"
if torch.cuda.is_available():
    __DEVICE__ = "cuda"

class Ours:
    def __init__(self, device):
        text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                projection_dim=1280,
                torch_dtype=torch.float16,
            )
            .eval()
            .requires_grad_(False)
        ) 

        tokenizer = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )

        prior = PriorTransformer.from_pretrained(
            "ECLIPSE-Community/ECLIPSE_KandinskyV22_Prior",
            torch_dtype=torch.float16,
        )

        self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
        ).to(device)

        self.pipe = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to(device)

    def inference(self, text, negative_text, steps, guidance_scale):
        gen_images = []
        for i in range(1):
            image_emb, negative_image_emb = self.pipe_prior(
                text, negative_prompt=negative_text
            ).to_tuple()
            image = self.pipe(
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            ).images
            gen_images.append(image[0])
        return gen_images


selected_model = Ours(device=__DEVICE__)


def get_images(text, negative_text, steps, guidance_scale):
    images = selected_model.inference(text, negative_text, steps, guidance_scale)
    new_images = []
    for img in images:
        new_images.append(img)
    return new_images[0]


with gr.Blocks() as demo:
    gr.Markdown(
        """<h1 style="text-align: center;"><b><i>ECLIPSE</i>: Revisiting the Text-to-Image Prior for Effecient Image Generation</b></h1>
        <h1 style='text-align: center;'><a href='https://eclipse-t2i.vercel.app/'>Project Page</a> | <a href='https://eclipse-t2i.vercel.app/'>Paper</a>  </h1>
        """
    )

    with gr.Group():
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )

        with gr.Row():
            with gr.Column():
                negative_text = gr.Textbox(
                    label="Enter your negative prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your negative prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )

        with gr.Row():
            steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=50, step=1)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=10, value=7.5, step=0.1
            )

        with gr.Row():
            btn = gr.Button(value="Generate Image", full_width=False)

    gallery = gr.Image(
        height=512, width=512, label="Generated images", show_label=True, elem_id="gallery"
    ).style(preview=False, columns=1)

    btn.click(
        get_images,
        inputs=[
            text,
            negative_text,
            steps,
            guidance_scale,
        ],
        outputs=gallery,
    )
    text.submit(
        get_images,
        inputs=[
            text,
            negative_text,
            steps,
            guidance_scale,
        ],
        outputs=gallery,
    )
    negative_text.submit(
        get_images,
        inputs=[
            text,
            negative_text,
            steps,
            guidance_scale,
        ],
        outputs=gallery,
    )

    with gr.Accordion(label="Ethics & Privacy", open=False):
        gr.HTML(
            """<div class="acknowledgments">
                <p><h4>Privacy</h4>
We do not collect any images or key data. This demo is designed with sole purpose of fun and reducing misuse of AI.
                <p><h4>Biases and content acknowledgment</h4>
This model will have the same biases as pre-trained CLIP model.               </div>
            """
        )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
