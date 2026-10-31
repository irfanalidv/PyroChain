"""PyroChain quick start: multimodal feature extraction with the default open model."""

from pyrochain import PyroChain, PyroChainConfig

chain = PyroChain(PyroChainConfig(device="cpu"))  # default model: microsoft/DialoGPT-small

product = {
    "title": "Wireless Bluetooth Headphones",
    "description": "Noise cancellation and 30-hour battery life.",
    "price": 199.99,
    "category": "electronics",
}
result = chain.extract_features(product, "Extract features for product recommendation")

features = result["features"][0]
print("modalities:", features["metadata"]["modalities"])
print("text features:", tuple(features["text_features"].shape))
print("validation score:", round(features["validation_score"], 2))
print(features["validation_feedback"])
