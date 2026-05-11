import sys
from pathlib import Path

qwen_repo = Path("/home/aditya/work/ml-stuff/tool/Qwen3-VL-Embedding")
sys.path.append(str(qwen_repo))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig
from PIL import Image
import os
import gradio as gr



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    quantization_config=bnb_config,
    device_map="auto",
)


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
image_paths = list(IMAGE_DIR.glob("*.jpg"))

image_inputs = [{"image": str(path)} for path in image_paths]

NUM_RESULTS = 5
MAX_RESULTS = 25

img_embeddings = []
imgs = []
p = 0

batch_size = 2


if os.path.exists(BASE_DIR / "qwen_image_embeddings.pt"):
    img_embeddings = torch.load(BASE_DIR / "qwen_image_embeddings.pt") 
else:
    for img_path in image_paths:
        imgs.append(img_path)
        p += 1

        print(f"Images queued : {p}/{len(image_paths)}", end="\r")

        if len(imgs) == batch_size:
            batch = [
                {"image": str(path)}
                for path in imgs
            ]

            with torch.no_grad():
                img_e = model.process(batch)
                img_e = F.normalize(img_e, p=2, dim=-1)

            img_embeddings.append(img_e.cpu())
            imgs = []

            torch.cuda.empty_cache()

    # leftover images
    if len(imgs) > 0:
        batch = [
            {"image": str(path)}
            for path in imgs
        ]

        with torch.no_grad():
            img_e = model.process(batch)
            img_e = F.normalize(img_e, p=2, dim=-1)

        img_embeddings.append(img_e.cpu())
        torch.cuda.empty_cache()

    img_embeddings = torch.cat(img_embeddings, dim=0)

    torch.save(img_embeddings, BASE_DIR / "qwen_image_embeddings.pt")


def search_image_paths(query: str, result_count: int) -> list[Path]:

    with torch.no_grad():
        query_embedding = model.process([
            {
                "text": query,
                "instruction": "Retrieve images that visually match the user's description."
            }
        ])
        query_embedding = F.normalize(query_embedding, p=2, dim=-1).cpu()

        scores = query_embedding @ img_embeddings.T

    result_count = max(1, min(int(result_count), len(image_paths)))
    top_results, indices = torch.topk(scores, result_count, dim=1)
    indices = indices.squeeze(0)

    out = [image_paths[i.item()] for i in indices]
    return out


def search_image_paths_from_image(query_image_path: str, result_count: int) -> list[Path]:
    
    with torch.no_grad():
        query_image_embedding = model.process([
            {
                "image": query_image_path,
                "instruction": "Retrieve images that visually match the given image."
            }
        ])
        query_image_embedding = F.normalize(query_image_embedding, p=2, dim=-1).cpu()

        scores = query_image_embedding @ img_embeddings.T

    result_count = max(1, min(int(result_count), len(image_paths)))
    top_results, indices = torch.topk(similarity, result_count, dim=1)
    indices = indices.squeeze(0)

    out = [image_paths[i.item()] for i in indices]
    return out


def run_search(query: str, query_image_path: str | None, result_count: int) -> tuple:
    """Gradio callback for the Search button and Enter key."""
    cleaned_query = query.strip()

    if not cleaned_query and not query_image_path:
        return tuple(
            gr.update(value=None, visible=False)
            for _ in range(MAX_RESULTS)
        )

    if query_image_path:
        result_paths = search_image_paths_from_image(query_image_path, result_count)
    else:
        result_paths = search_image_paths(cleaned_query, result_count)

    results = [
        str(path) if path is not None else None
        for path in result_paths
        if path is not None
    ]
    return tuple(
        gr.update(
            value=results[index] if index < len(results) else None,
            visible=index < len(results),
        )
        for index in range(MAX_RESULTS)
    )


with gr.Blocks(
    title="Image Search",
    css="""
    .results-list {
        max-width: 380px;
    }
    .result-image {
        max-width: 340px;
    }
    """,
) as demo:
    gr.Markdown("# Image Search")

    with gr.Row():
        query_input = gr.Textbox(
            label="Search query",
            placeholder="Enter a sentence...",
            scale=5,
        )
        search_button = gr.Button("Search", variant="primary", scale=1)

    query_image_input = gr.Image(
        label="Query image",
        type="filepath",
        sources=["upload"],
        width=320,
    )

    result_count_slider = gr.Slider(
        minimum=1,
        maximum=MAX_RESULTS,
        value=5,
        step=1,
        label="Images to show",
    )

    with gr.Column(elem_classes="results-list"):
        result_images = [
            gr.Image(
                label=f"Result {index + 1}",
                type="filepath",
                height=240,
                width=320,
                visible=False,
                buttons=[],
                elem_classes="result-image",
            )
            for index in range(MAX_RESULTS)
        ]

    search_button.click(
        fn=run_search,
        inputs=[query_input, query_image_input, result_count_slider],
        outputs=result_images,
    )
    query_input.submit(
        fn=run_search,
        inputs=[query_input, query_image_input, result_count_slider],
        outputs=result_images,
    )


if __name__ == "__main__":
    demo.launch(share= True)
