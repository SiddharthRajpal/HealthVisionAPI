from flask import Flask, request
import imagerec
import base64
import io
from PIL import Image

app = Flask(__name__)


@app.route('/braintumor', methods=['POST','PUT'])
def BrainTumor():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)


