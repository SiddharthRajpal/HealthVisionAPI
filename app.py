from flask import Flask, request
import imagerec
import base64
from PIL import Image

app = Flask(__name__)
@app.route('/braintumorbase', methods=['POST', 'PUT'])
def BrainTumorBase():
    try:
        image_data = request.get_data()
        image_data = base64.b64decode(image_data)
        
        # Convert the image data to PIL Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Perform image recognition
        pred, _ = imagerec.imagerecognise(image_array, "Models/BrainTumuorModel.h5", labelpath="Models/BrainTumuorLabels.txt")
        
        # Return the prediction as a response
        response = {
            'prediction': pred
        }
        
        return response
    except Exception as e:
        return str(e)
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

