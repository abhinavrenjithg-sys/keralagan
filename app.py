"""KeralaGAN demo: upload a photo, pick a style.

Runs locally (python app.py) or as a Hugging Face Space (see scripts/build_space.py).
Looks for exported generators in weights/:
    weights/mural_G_AB.pt   weights/monsoon_G_AB.pt   weights/night_G_AB.pt
"""

import os
from functools import lru_cache

import gradio as gr
import torch

from keralagan.translate import load_generator, translate_image

WEIGHTS = os.environ.get("KERALAGAN_WEIGHTS", "weights")
STYLES = {
    "Kerala mural": "mural_G_AB.pt",
    "Monsoon": "monsoon_G_AB.pt",
    "Night": "night_G_AB.pt",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def available_styles():
    return [s for s, f in STYLES.items() if os.path.exists(os.path.join(WEIGHTS, f))]


@lru_cache(maxsize=3)
def get_model(style):
    G, cfg = load_generator(os.path.join(WEIGHTS, STYLES[style]), device=DEVICE)
    return G, cfg


def run(image, style, resolution):
    if image is None:
        raise gr.Error("Upload a photo first.")
    if style not in available_styles():
        raise gr.Error(f"The '{style}' model isn't trained yet. Try another style.")
    G, _ = get_model(style)
    return translate_image(G, image, size=int(resolution), max_side=1024, device=DEVICE)


ready = available_styles()
with gr.Blocks(title="KeralaGAN") as demo:
    gr.Markdown(
        "# KeralaGAN\n"
        "Repaint a photo as a **Kerala mural**, in **monsoon** weather, or **at night**. "
        "A CycleGAN trained on unpaired images. "
        + ("" if ready else "\n\n**No trained weights found in `weights/` yet.** Train a model and export it first.")
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Your photo")
            style = gr.Radio(list(STYLES), value=(ready[0] if ready else "Kerala mural"), label="Style")
            res = gr.Slider(128, 512, value=256, step=32, label="Resolution (short side)",
                            info="The model was trained at 256. Higher keeps more detail but drifts in style.")
            btn = gr.Button("Translate", variant="primary")
        out = gr.Image(type="pil", label="KeralaGAN output")
    btn.click(run, [inp, style, res], out)
    if os.path.isdir("examples"):
        ex = sorted(os.path.join("examples", f) for f in os.listdir("examples")
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if ex:
            gr.Examples([[e, ready[0] if ready else "Kerala mural", 256] for e in ex], [inp, style, res])

if __name__ == "__main__":
    demo.launch()
