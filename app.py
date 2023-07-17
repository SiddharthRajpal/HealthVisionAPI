from flask import Flask, request
import imagerec

app = Flask(__name__)

@app.route('/braintumor', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)

@app.route('/covid', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)


@app.route('/glaucoma', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)

@app.route('/pneumonia', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)


@app.route('/skincancer', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)

@app.route('/tuberculosis', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"BrainTumuorModel.h5",labelpath="BrainTumuorLabels.txt")

    return str(pred)

