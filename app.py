from flask import Flask
import io
from flask import send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from PIL import Image


app = Flask(__name__)


# Note 1: We need this loss function here to reconstruct the model.
def loss (y_true, y_pred):
    """ This loss computes the squared difference between the predicted and
        true value; averaged across the batch.
    """
    result     = (y_true - y_pred) ** 2
    loss_batch = tf.math.reduce_sum(result, axis=-1)
    mean_loss  = tf.math.reduce_mean(loss_batch)
    return mean_loss


model_path = "logs/20200311-203307/ckpt/weights.03-0.02.hdf5"
net        = tf.keras.models.load_model( model_path
                                       , custom_objects = {"loss": loss} )


# Note 2: Pick out the z-dim from the network, and let's just
#         display the network for good measure.
z_dim = net.get_layer("z").output.shape[-1]

net.summary()


# Note 3: 
# In TF2/Keras it's a bit annoying to sample from the model. If we
# want to run a $z$ vector through, we need to build the output from
# a new input. So we just pick out all layers after the $z$ layer, and
# then apply them to our input.
in_z   = tf.keras.Input(shape=(10))
layers = net.layers[6:] # Layer 6 is the one after the "z" layer.
output = in_z
for l in layers:
    output = l(output)

sample_model = Model( in_z, outputs=output )
sample_model.summary()


@app.route("/")
def index ():
    return "Visit <a href='/sample/312'>/sample/312</a>."


@app.route("/sample/<int:seed>")
def sample (seed=2):

    # Note 4: We can just pick a random z vector now; and push it through the
    # network, then get out an image, and render it.
    np.random.seed(seed)
    z_vect = np.random.uniform(-1, 1, (1, z_dim))


    img = sample_model.predict(z_vect)
    img = Image.fromarray(img.squeeze() * 255)
    img = img.resize((200, 200), Image.ANTIALIAS)

    sprite = Image.new(mode="L", size=(200, 200))
    sprite.paste(img)

    arr = io.BytesIO()
    # Annoyingly, there is a bug in flask that means we need to write to
    # a tmp file. If flask was well-behaved, we could serve the image directly
    # from the `BytesIO` object.
    sprite.save("/tmp/img.jpg", format="jpeg")

    resp = send_file("/tmp/img.jpg", mimetype='image/jpeg')

    return resp
