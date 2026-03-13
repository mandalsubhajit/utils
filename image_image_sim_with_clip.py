from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/path/to/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 2. Load and preprocess your image
image = Image.open("/path/to/image.jpg").convert("RGB")
# inputs = processor(images=image, return_tensors="pt")
inputs = processor(
		text = None,
		images = image,
		return_tensors="pt"
		)["pixel_values"].to(device)

# 3. Extract the visual features
with torch.no_grad():
    outputs = model.get_image_features(inputs)
    image_features = outputs['last_hidden_state'].cpu().detach().numpy().reshape(1, -1)

# 4. Normalize the embedding (crucial for similarity matching)
# image_embeddings = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# image_embeddings = image_embeddings.view(1, -1)

# 5. Calculate Similarity - Assuming emb1 and emb2 the different embeddings (=image_features)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(emb1, emb2)
print(f"Similarity Score: {similarity:.4f}")
# 1.0 is identical, 0.0 is completely different
