import gradio as gr
from main import TextToLayoutPipeline
from PIL import Image
import os

# initialize the layout pipeline once
enabled_pipeline = TextToLayoutPipeline()

def generate_layout(user_text: str):
    # run the pipeline to generate and save the layout image
    enabled_pipeline.run(user_text=user_text)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_poster.png")
    # load the generated image and return it for display
    return Image.open(output_path)

if __name__ == "__main__":
    # create and launch the Gradio interface
    iface = gr.Interface(
        fn=generate_layout,
        inputs=gr.Textbox(lines=4, placeholder="Enter layout description...", label="Prompt"),
        outputs=gr.Image(type="pil", label="Generated Layout"),
        title="Text-to-Layout Generator",
        description="Enter a description of the layout you want, and generate a poster image."
    )
    iface.launch(share=True)