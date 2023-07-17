from flask import Flask, request
import imagerec

app = Flask(__name__)

@app.route('/braintumor', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/covid', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/CovidModel.h5",labelpath="Models/CovidLabels.txt")

    return str(pred)


@app.route('/glaucoma', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/GlaucomaModel.h5",labelpath="Models/GlaucomaLabels.txt")

    return str(pred)

@app.route('/pneumonia', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/Pnemonia.h5",labelpath="Models/labelsPnemonia.txt")

    return str(pred)


@app.route('/skincancer', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/SkinCancerModel.h5",labelpath="SkinCancerLabel.txt")

    return str(pred)

@app.route('/tuberculosis', methods=['POST'])
def callback():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/tuberculosis_model.h5",labelpath="tb_labels.txt")

    return str(pred)

