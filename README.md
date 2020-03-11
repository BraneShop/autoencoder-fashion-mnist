This is a little hacked-together notebook demonstrating how to plot an
arbitrary tensor in the "Projector" pane of TensorBoard, with images for the
points.

It's super hacked-together because TensorFlow 2.0 has "simplified" the API for
plotting embeddings, and _only_ plots `Embedding` layers by default.

See [this issue](https://github.com/tensorflow/tensorboard/issues/2471) for
when this feature might be cleaned up.

## Setup/Running

1. Clone the repo

2. Create a conda environment, `conda create -n autoencoder-fashion-mnist python=3`

3. Upgrade pip: `pip install pip --upgrade`

4. Install requirements: `pip install -r requirements.txt`

5. Run the notebook: `jupyter notebook` ... !

6. Run TensorBoard with `tensorboard --logdir logs`

7. Jump over to the "Projector" tab!
