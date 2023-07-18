from flask import Flask, request
import imagerec
import base64
import io
from PIL import Image

app = Flask(__name__)
@app.route('/braintumorbase', methods=['POST', 'PUT'])
def BrainTumorBase():
    def callback():
        image_file = request.files['image']
        pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

        return str(pred)


@app.route('/braintumor', methods=['POST','PUT'])
def BrainTumor():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)
@app.route('/covid', methods=['POST'])
def Covid():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/glaucoma', methods=['POST'])
def Glaucoma():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/pneumonia', methods=['POST'])
def Pnemonia():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)


@app.route('/skincancer', methods=['POST'])
def SkinCancer():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/tuberculosis', methods=['POST'])
def Tuberculosis():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

