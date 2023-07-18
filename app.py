from flask import Flask, request
import imagerec
import base64

app = Flask(__name__)
@app.route('/braintumorbase', methods=['POST','PUT'])
def BrainTumorBase():
    image_data = request.get_data()
    padding = len(image_data) % 4
    if padding > 0:
        image_data += '=' * (4 - padding)
    image_file = base64.b64decode(image_data)
    image_utf = image_file.decode('utf-8')
    print(image_file)

    print(image_utf)
    try:
        pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")
    except:
        pred,con = imagerec.imagerecognise(image_utf,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/braintumor', methods=['POST','PUT'])
def BrainTumor():
    image_file = request.files['image']

    print(image_file)
    pred,con = imagerec.imagerecognise(image_file,"Models/BrainTumuorModel.h5",labelpath="Models/BrainTumuorLabels.txt")

    return str(pred)

@app.route('/covid', methods=['POST'])
def Covid():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/CovidModel.h5",labelpath="Models/CovidLabels.txt")

    return str(pred)


@app.route('/glaucoma', methods=['POST'])
def Glaucoma():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/GlaucomaModel.h5",labelpath="Models/GlaucomaLabels.txt")

    return str(pred)

@app.route('/pneumonia', methods=['POST'])
def Pnemonia():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/Pnemonia.h5",labelpath="Models/labelsPnemonia.txt")

    return str(pred)


@app.route('/skincancer', methods=['POST'])
def SkinCancer():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/SkinCancerModel.h5","Models/SkinCancerLabel.txt")

    return str(pred)

@app.route('/tuberculosis', methods=['POST'])
def Tuberculosis():
    image_file = request.files['image']
    pred,con = imagerec.imagerecognise(image_file,"Models/tuberculosis_model.h5",labelpath="tb_labels.txt")

    return str(pred)

