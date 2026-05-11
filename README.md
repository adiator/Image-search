# Image Search

This project is a multimodal image retrieval app built with Gradio and
Qwen3-VL-Embedding. It lets you search a local image collection using
either a text query or an uploaded image and returns the top matching
results in the browser.

This app uses Qwen's official `Qwen3-VL-Embedding` repository. The search UI lives in
[`app.py`](./app.py), but the embedding model implementation is imported
from a separately cloned copy of the Qwen repository.

The two repos can live anywhere on disk. The app will first look for a
`QWEN3_VL_EMBEDDING_REPO` environment variable, and if that is not set,
it will look for `Qwen3-VL-Embedding` as a sibling directory next to the
image search repo.

## Setup

1. Clone this image search repository:

```bash
git clone https://github.com/adiator/Image-search.git
cd Image-search
```

2. Clone the Qwen repository into the same parent directory as
   `Image-search`:

```bash
cd ..
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
```

3. Create and populate the Qwen environment using its setup script:

```bash
cd Qwen3-VL-Embedding
bash scripts/setup_environment.sh
source .venv/bin/activate
```

That repo-managed environment installs the main model stack, including
`torch`, `torchvision`, `transformers`, and `qwen-vl-utils`.

4. Install the extra packages used by this Gradio app:

```bash
cd Image-search/imagesearch
uv pip install -r requirements.txt
```

5. Tell the app where your Qwen repo lives.

You can do that in either of these ways:

- Set an environment variable:

```bash
export QWEN3_VL_EMBEDDING_REPO=/path/to/Qwen3-VL-Embedding
```

- Or rely on the default layout and place the cloned Qwen repo here
  relative to the image search repo:
  - `../Qwen3-VL-Embedding`

## Running the App

Once the Qwen repo environment is active, start the app with:

```bash
cd ../Image-search/imagesearch
python app.py
```

On first launch, the app computes image embeddings for the local
`images/` folder and stores them in `qwen_image_embeddings.pt`. Later
launches reuse that cached tensor.

## Notes

- This project currently uses the embedding model only.
- The reranker is not wired into the app yet.
- The current app code uses `BitsAndBytesConfig` for 4-bit loading, so
  `bitsandbytes` must be installed in the active environment.
