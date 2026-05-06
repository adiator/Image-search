import torch, open_clip
from PIL import Image
from pathlib import Path
import gradio as gr
import os

model_name = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

model, preprocess = open_clip.create_model_from_pretrained(
    f"hf-hub:{model_name}"
)
tokenizer = open_clip.get_tokenizer(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
NUM_RESULTS = 5
MAX_RESULTS = 25

batch_size = 32
image_paths = list(IMAGE_DIR.glob("*.jpg"))

if os.path.exists(BASE_DIR / "embeddings.pt"):
    img_embeddings = torch.load(BASE_DIR / "embeddings.pt") 
else:
    imgs = []
    img_embeddings = []

    p = 0
    for img_path in image_paths:
        image = preprocess(Image.open(img_path)).to(device) #([3, 224, 224])
        p+=1
        imgs.append(image)

        print(f"Images processed : {p}/{len(image_paths)}", end='\r')
        if(len(imgs) == batch_size):
            with torch.no_grad():
                image_batch = torch.stack(imgs).to(device) #([32, 3, 224, 224]
                img_e = model.encode_image(image_batch) #([32, 512]
                img_e /= img_e.norm(dim=-1, keepdim=True)
                img_embeddings.append(img_e)
            imgs = []

        
    if len(imgs) > 0:
        with torch.no_grad():
                image_batch = torch.stack(imgs).to(device)
                img_e = model.encode_image(image_batch)
                img_e /= img_e.norm(dim=-1, keepdim=True)
                img_embeddings.append(img_e)

    img_embeddings = torch.cat(img_embeddings, dim=0) #([3000, 512]
    torch.save(img_embeddings, BASE_DIR / "embeddings.pt")


def search_image_paths(query: str, result_count: int) -> list[Path]:

    with torch.no_grad():
        text = tokenizer(query).to(device)
        text_e = model.encode_text(text)
        text_e /= text_e.norm(dim=-1, keepdim=True)
        similarity = img_embeddings @ text_e.T
        similarity = similarity.squeeze(1)

    result_count = max(1, min(int(result_count), len(image_paths)))
    top_results, indices = torch.topk(similarity, result_count, dim=0)

    out = [image_paths[i.item()] for i in indices]
    return out


def search_image_paths_from_image(query_image_path: str, result_count: int) -> list[Path]:
    """Return matching image paths for an uploaded query image.

    Fill this in with your image-to-image retrieval logic.

    Suggested shape:
    1. Load the uploaded image from `query_image_path`.
    2. Preprocess it with the same CLIP preprocessing pipeline.
    3. Encode it with `model.encode_image(...)`.
    4. Compare that embedding against `img_embeddings`.
    5. Return the top `result_count` paths from `image_paths`.
    """
    image = preprocess(Image.open(query_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_e = model.encode_image(image)
        img_e /= img_e.norm(dim=-1, keepdim=True)
        similarity = img_embeddings @ img_e.T
        similarity = similarity.squeeze(1)

    result_count = max(1, min(int(result_count), len(image_paths)))
    top_results, indices = torch.topk(similarity, result_count, dim=0)

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
