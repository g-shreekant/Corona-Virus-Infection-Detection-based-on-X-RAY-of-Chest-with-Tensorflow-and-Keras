from flask import Flask, render_template, request
from io import BytesIO
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import load_model
import os
import PIL
from PIL import Image
import numpy as np
from base64 import b64encode
import tensorflow as tf
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'

bootstrap = Bootstrap(app)

saved_model = tf.keras.models.load_model("covid19.model")
saved_model._make_predict_function()

class UploadForm(FlaskForm):
    photo = FileField('Upload an image',validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')

def preprocess(image,target_size):
    # width, height = img.shape[0], img.shape[1]
    # img = image.array_to_img(img, scale=False)

    # desired_width, desired_height = 224, 224

    # if width < desired_width:
    #  desired_width = width
    # start_x = np.maximum(0, int((width-desired_width)/2))

    # img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    # img = img.resize((224, 224, 3))

    # img = image.img_to_array(img)
    # return img / 255.

    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image



@app.route('/', methods=['GET','POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
     print(form.photo.data)
     
     #img = image.img_to_array(original_img)
     #img = np.expand_dims(img, axis=0)

     image_stream = form.photo.data.stream
     original_img = Image.open(image_stream)
     img = preprocess(original_img,target_size=(224,224))



     prediction = saved_model.predict(img)

     if (prediction[0][0]>prediction[0][1]):
       result = "yes"
     else:
       result = "no"

     byteIO = BytesIO()
     original_img.save(byteIO, format=original_img.format)
     byteArr = byteIO.getvalue()
     encoded = b64encode(byteArr)

     return render_template('result.html', result=result, encoded_photo=encoded.decode('ascii'))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)