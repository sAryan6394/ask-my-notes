from sentence_transformers import SentenceTransformer
import numpy as np

# Loads a small, free, pretrained model that converts text into vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The mitochondria is the powerhouse of the cell",
    "Cell organelles produce energy for the cell",
    "I want to eat pizza tonight"
]

# Turn each sentence into a vector (a list of numbers representing its meaning)
embeddings = model.encode(sentences)

print("Shape of each embedding:", embeddings[0].shape)

# Manually compute cosine similarity between sentence 1 and sentence 2
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print("Similarity between sentence 1 and 2 (both about cells):", sim_1_2)
print("Similarity between sentence 1 and 3 (unrelated):", sim_1_3)