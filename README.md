This is a little hacked-together notebook demonstrating how to plot an
arbitrary tensor in the "Projector" pane of TensorBoard, with images for the
points.

It's super hacked-together because TensorFlow 2.0 has "simplified" the API for
plotting embeddings, and _only_ plots `Embedding` layers by default.

See [this issue](https://github.com/tensorflow/tensorboard/issues/2471) for
when this feature might be cleaned up.
