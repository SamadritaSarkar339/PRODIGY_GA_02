import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate image from prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=1, placeholder="Enter your prompt here...", label="Prompt"),
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Text-to-Image Generator",
    description="Enter a prompt and generate an AI image using Stable Diffusion.",
)

# Launch the UI
interface.launch()
