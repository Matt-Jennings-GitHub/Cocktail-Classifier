 # Modules
import os
from sklearn.utils import shuffle
import cv2
from pathlib import Path
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model, model_from_json
from keras.callbacks import TensorBoard
from tkinter import *
from tkinter.filedialog import askopenfilename
import PIL
from PIL import ImageTk

# Variables
classes = ['beer','bloody_mary','cosmopolitan','espresso_martini','gin_and_tonic','manhattan','margarita','martini','mojito','old_fashioned','rum_and_coke','pina_colada','screwdriver']

reprocess = False
data_path = 'Training Data'
target_size = (75, 75)
padding_colour = (255,255,255) # Set 'crop' to crop to square, 'stretch' for resize only
crop = False
train_fraction = 0.8

retrain = False
epochs = 20
batch_size = 32

# Preprocess Images
def preprocess_image(input_path, target_size, padding_colour): # Returns np array of processed image
    # Input Image
    img = cv2.imread(input_path) # cv2 import as BGR np array
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    except:
        return 'Invalid' # Return False for invalid file

    w, h = img.shape[1], img.shape[0]

    # Add padding to make square
    if padding_colour != 'crop' and padding_colour != 'stretch':
        if w > h:
            img = cv2.copyMakeBorder(img, int((w - h) / 2), int((w - h) / 2), 0, 0, cv2.BORDER_CONSTANT, value=padding_colour)
        elif h > w:
            img = cv2.copyMakeBorder(img, 0, 0, int((h - w) / 2), int((h - w) / 2), cv2.BORDER_CONSTANT, value=padding_colour)

    # Crop to square
    elif padding_colour == 'crop':
        if w > h :
            img = img[0:h, int((w-h)/2):w-int((w-h)/2)]
        elif h > w :
            img = img[int((h - w) / 2):h - int((h - w) / 2), 0:w]

    # Rescale to target size
    img = cv2.resize(img, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)

    return img

# Process images
def process_images():
    # Clear existing processed images
    for class_name in classes:
        for img_name in os.listdir('{}/Processed/{}'.format(data_path, class_name)):
                os.remove('{}/Processed/{}/{}'.format(data_path, class_name, img_name))

    # Preprocess images
    for class_name in classes:
        print(class_name)
        for img_name in os.listdir('{}/Source/{}'.format(data_path,class_name)):
            print(img_name)
            img = preprocess_image('{}/Source/{}/{}'.format(data_path, class_name, img_name), target_size, padding_colour)

            # Save processed Images
            if img != 'Invalid': # Check for invalid file
                img = PIL.Image.fromarray(img, 'RGB')
                img.save('{}/Processed/{}/{}.png'.format(data_path, class_name, img_name.split('.')[0]), 'PNG')
    return

if reprocess:
    process_images()

# Train Network
def train_network(epochs, batch_size):
    model.fit(x_train_features, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test_features, y_test), shuffle=True, callbacks=[logger])
    results = model.evaluate(x_test_features, y_test, verbose=0)
    print('Validation Loss: {} Validation Accuracy: {}'.format(results[0], results[1]))

    # Save Model
    model_structure = model.to_json()
    f = Path("model_structure.json")
    f.write_text(model_structure)
    model.save_weights("model_weights.h5")
    return

if retrain :
    # Load processed images
    images = []
    labels = []
    for class_name in classes:
        for img_name in os.listdir('{}/Processed/{}'.format(data_path, class_name)):
            img = cv2.imread('{}/Processed/{}/{}'.format(data_path, class_name, img_name))

            # Form input and label arrays
            images.append(img)
            labels.append(classes.index(class_name))

    images, labels = shuffle(images, labels) # Simultaneous shuffle
    x_train, x_test = np.array(images[0:int(len(images)*train_fraction)]), np.array(images[int(len(images)*train_fraction):-1]) # Train Test split
    y_train, y_test = np.array(labels[0:int(len(labels)*train_fraction)]), np.array(labels[int(len(labels)*train_fraction):-1])
    y_train, y_test = keras.utils.to_categorical(y_train, len(classes)), keras.utils.to_categorical(y_test, len(classes)) # One hot encode classes

    x_train, x_test = vgg16.preprocess_input(x_train), vgg16.preprocess_input(x_test) # VGG16 Normalise

    # Define Network
    # Feature Extractor
    pretrained_network = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
    x_train_features, x_test_features = pretrained_network.predict(x_train), pretrained_network.predict(x_test)  # Extract Features

    # Main Model
    logger = TensorBoard(log_dir='logs', write_graph=True)

    model = Sequential()

    model.add(Flatten(input_shape=x_train_features.shape[1:], name='Flatten_Layer'))

    model.add(Dense(256, activation='relu', name='Final_Hidden_Layer'))
    model.add(Dropout(0.3))

    model.add(Dense(len(classes), activation='softmax', name='Output_Layer'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train Network
    train_network(epochs, batch_size)

# Load Model
pretrained_network = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

# User Interface

# Load Image File
def load_image():
    try:
        img_path = askopenfilename(filetypes=[("JPG, JPEG or PNG", "*.jpg; *.png; *.jpeg")])
        print(img_path)

        # Update Display
        img_name = img_path.split("\\")[-1].split("/")[-1]
        InfoLabel.configure(text=img_name)

        img = PIL.Image.open(img_path)
        img.thumbnail((260,260), PIL.Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        ImageLabel.configure(image=img)
        ImageLabel.image = img
    except:
        print('Error loading image.')

    classify_image(img_path)

# Test Image
def classify_image(img_path):
    # Input Test Image
    img = preprocess_image(img_path, target_size, padding_colour)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    # Prediction
    features = pretrained_network.predict(img) # Extract Features
    results = model.predict(features)[0] # Predict

    # Display results
    print("----{}----".format(img_path))
    results = list(results.flatten())
    results = sorted(zip(classes, results), key=lambda i: i[1], reverse=True)
    for pair in results:
        print('{}: {:.4f}'.format(pair[0], pair[1]))

    # Update Display
    for i in range(0, num_classes) :
        ClassLabels[i].configure(text='{}: {:.4f}'.format(results[i][0], results[i][1]))

# GUI

# Window
num_classes = 5
window_width = "305"
window_height = str(340 + num_classes*20)


window = Tk()
window.resizable(0, 0)
window.pack_propagate(0)
window.geometry("{}x{}".format(window_width,window_height))
window.title("Cocktail Classifier")

# Define Objects
LoadButton = Button(text="Load",height=1,width=36,command=load_image)
InfoLabel = Label(text="Upload an Image")
img = PIL.Image.open("loadicon.png")
img.thumbnail((260,260), PIL.Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
ImageLabel = Label(image=img,width=260,height=260)
ClassLabels = []
for i in range(0, num_classes):
    ClassLabels.append(Label(text=''))

# Place Objects
LoadButton.place(x=20, y=10)
InfoLabel.place(x=20, y=40)
ImageLabel.place(x=20, y=60)
for i in range(0, num_classes):
    ClassLabels[i].place(x=20, y=330 + i * 20)

window.mainloop()















