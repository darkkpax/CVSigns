from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
model = YOLO(str(MODEL_PATH))


def predict(image: Image.Image, conf: float, iou: float):
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    plotted = results[0].plot()
    return plotted


with gr.Blocks(title="Russian Traffic Signs YOLO12") as demo:
    gr.Markdown(
        """
        # Russian Traffic Signs YOLO12
        Загрузите изображение, и модель определит дорожные знаки.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Изображение")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
            iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
            run_btn = gr.Button("Распознать")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Результат")

    run_btn.click(
        fn=predict,
        inputs=[input_image, conf, iou],
        outputs=[output_image],
    )


if __name__ == "__main__":
    demo.launch()
