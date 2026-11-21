"""Content side of the recommender: what a movie is *about*, as text and as a sentence embedding.

Used for natural-language ("semantic") search and for "similar by content" in the app. Everything here
runs without PyTorch: the sentence encoder is an ONNX model executed by ONNX Runtime.
"""
