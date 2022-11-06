from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
from PIL import Image

try:
	import shutil
	shutil.rmtree('uploaded/image')
	# os.system('cd uploaded')
	os.system('mkdir uploaded/image')
	os.system('cd ..')
	print()
except:
	print('fail')
	pass

model = tf.keras.models.load_model('model_06-0.91.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/image'

@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds():
	vals = ['Star', 'Galaxy']
	a = os.listdir('uploaded/image')
	b = 'uploaded/image/' + a[0]
	print(b)
	image = Image.open(b)
	numpydata = np.asarray(image)

	X = numpydata / 255.0
	X = X.reshape(1, X.shape[0], X.shape[1], 3)

	pred = model.predict_generator(X)
	print(pred)
	# print(np.argmax(pred))
	return vals[int(pred[0][0])]

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds()
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run()
