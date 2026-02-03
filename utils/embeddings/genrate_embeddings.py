import torch
import clip
import pickle
import os


def generate_clip_embedding(text, model, device):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def create_custom_task_embeddings_clip(task_names, output_path="custom_task_clip_embeddings.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _ = clip.load("ViT-B/32", device=device)
    print(f"CLIP model loaded. Text embedding dimension: {model.text_projection.shape[1]}")
    print(f"Generating CLIP embeddings for {len(task_names)} custom tasks...")

    task_embeddings = {}
    for task_name in task_names:
        print(f"Processing task: {task_name}")
        embedding = generate_clip_embedding(task_name, model, device)
        task_embeddings[task_name] = embedding
        print(f"  - Embedding shape: {embedding.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(task_embeddings, f)

    print(f"\nEmbeddings saved to: {output_path}")
    print(f"Task names: {list(task_embeddings.keys())}")
    return task_embeddings


def create_embeddings_from_dataset_dir(data_dir, output_path=None):
    files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
    if not files:
        raise ValueError(f"No .hdf5 files found in: {data_dir}")

    task_names = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.endswith("_demo"):
            name = name[:-5]
        task_names.append(name)

    if output_path is None:
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        output_path = os.path.join(os.path.dirname(__file__), f"{dataset_name}.pkl")

    return create_custom_task_embeddings_clip(task_names, output_path=output_path)
