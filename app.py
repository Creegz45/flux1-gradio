import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "black-forest-labs/flux.1-dev"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(fn=generate, inputs="text", outputs="image", title="Flux.1-dev Diffuser").launch(server_name="0.0.0.0", server_port=7860)
