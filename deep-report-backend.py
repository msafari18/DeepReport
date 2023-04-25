import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import skimage
from load_model import load_transformer
from DeepReport_model.vocabulary import Vocabulary
import json
from random import randrange
from DeepReport_model.segmentation import segmentation_model


app = Flask(__name__)

# @app.route('/img/<path:filename>')
# def send_file(filename):
#     return send_from_directory("img/",filename)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    # send_file("logo.png")
    if request.method == 'POST':
        first_name = request.form.get('fname')
        last_name = request.form.get('lname')
        id = request.form.get('id')
        file = request.files['file']
       # print(request.form)
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        name = first_name + " " + last_name
        return redirect(url_for('prediction', filename=filename, name = name, id = id))
    return render_template('index.html')


@app.route('/prediction/<filename>/<name>/<id>')
def prediction(filename, name, id):

    path = "DeepReport_model/Patient_info"
    original_image_path = os.path.join(path, "uploaded_image/" + str(id) + ".jpg")
    segmented_image_path = os.path.join(path, "segmented_image/" + str(id) + ".jpg")

    # Getting data
    img = skimage.io.imread(os.path.join('uploads', filename))

    #Loading transformer and get report
    report = load_transformer(img, vocab)

    #segmentaion
    segmentation_model(img, segmented_image_path)

    #Creating the dictionary
    predictions = {"id": id, "report": report, "name": name, "img_path": original_image_path,
                   "segmentation_result_path": segmented_image_path, "ImageID": "IMG"+str(id[1:]), "ReportID": "RE"+str(id[1:])}

    #write in file
    skimage.io.imsave(original_image_path, img)
    json_object = json.dumps(predictions, indent=4)
    with open(path+"/patient_data.json", "w") as outfile:
        outfile.write(json_object)

    # return results
    return render_template('predict.html', predictions=predictions)

if __name__ == '__main__':
    global vocab
    vocab = Vocabulary(vocab_file='DeepReport_model/data/vocab.pkl')
    app.run(host='0.0.0.0', port=8080)

