# %% [markdown]
# Importing MobileNetV2

# %%
from tensorflow.keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Dense
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import preprocess_input
from skimage.transform import resize
from imageio import imread
import numpy as np
import pandas as pd
from keras.applications.mobilenet_v2 import MobileNetV2

# %%
model = MobileNetV2(weights='imagenet')

# %% [markdown]
# ### Data Preprocessing:

# %% [markdown]
# Loading the data and generating labels

# %%

# %%
# Read the csv file into the DataFrame
df = pd.read_csv('Labelled_Data.csv')

# Get the number of rows in the DataFrame
num_of_images = len(df)

# Print the number of images
print(f"Number of Labelled Images: {num_of_images}")

# %%
# num_of_images is the number of images that I want to train
data = np.empty((num_of_images, 224, 224, 3))

for i in range(num_of_images):
    image_name = df.iloc[i, 1]
    im = imread(f'Training-Images/{image_name}')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i] = im


# %% [markdown]
# Tidy up the data and create the matrix

# %% [markdown]
# |Categories | Number Representation |
# | --- | --- |
# | Cosmetics and Medicine | 0 |
# | Electronics | 1 |
# | Textile, Clothing & Shoes | 2 |
# | Furniture & Home appliances | 3 |
# | Toys  | 4 |
# | Books & Movies | 5 |
#

# %%
"""
Generating Labels
"""
category_map = {
    "Cosmetics & Medicine": 0,
    "Electronics": 1,
    "Textile, Clothing & Shoes": 2,
    "Furniture & Home appliances": 3,
    "Toys ": 4,
    "Books & Movies": 5
}


labels = np.empty(num_of_images, dtype=int)

for index, row in df.iterrows():
    category = row[2]
    labels[index] = category_map[category]


# %% [markdown]
# Generating predictions for items, without transfer learning

# %%

# %%
predictions = model.predict(data)

for decoded_prediction in decode_predictions(predictions, top=1):
    for name, desc, score in decoded_prediction:
        print('- {} ({:.2f}%)'.format(desc, 100*score))

# %% [markdown]
# Modifying the Model

# %%

# %%
model.summary()

# %%
label_output = Dense(6, activation='softmax')
# this is a keras tensor, to connect the nodes to our putput nodes
label_output = label_output(model.layers[-2].output)
label_input = model.input
label_model = Model(inputs=label_input, outputs=label_output)

for layer in label_model.layers[:-1]:
    layer.trainable = False

# %% [markdown]
# Compiling the Model

# %%
label_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# %% [markdown]
# Training the Model

# %%
label_model.fit(
    x=data,
    y=labels,
    epochs=20,
    verbose=2
)

# %% [markdown]
# Generating predictions for training data

# %%
predictions = label_model.predict(data)

# %%
predictions.shape

# %%
np.argmax(predictions, axis=1)

# %% [markdown]
# Generating separate training and validation sets

# %%
# separate training and validation sets randomly. 80% training, 20% validation

X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape


# %%
label_model.fit(
    x=X_train,
    y=y_train,
    epochs=20,
    verbose=2,
    validation_data=(X_val, y_val)
)

predictions = label_model.predict(X_val)

predictions.shape


# %%
np.argmax(predictions, axis=1)


# %%
# now, check the percentage of accuracy of the model

accuracy_score(y_val, np.argmax(predictions, axis=1))


# %%

print("Image Classification using MobileNetV2")

# Print the number of images
print(f"Number of Labelled Images: {num_of_images}")

# Print Summary of the model
label_model.summary()

# Print the accuracy of the model
print(
    f"Accuracy of the model: {accuracy_score(y_val, np.argmax(predictions, axis=1))}")

# Overview of the results:
print("Overview of the results:")
for i in range(len(predictions)):
    print(
        f"Image {i+1}: Actual label: {y_val[i]}, Predicted label: {np.argmax(predictions[i])}")


# Print the f-measure of the model
print(
    f"F-measure of the model: {f1_score(y_val, np.argmax(predictions, axis=1), average='macro')}")

# Print the confusion matrix of the model
print(
    f"Confusion Matrix of the model: {confusion_matrix(y_val, np.argmax(predictions, axis=1))}")

# Print the classification report of the model
print(
    f"Classification Report of the model: {classification_report(y_val, np.argmax(predictions, axis=1))}")

# Print the precision of the model
print(
    f"Precision of the model: {precision_score(y_val, np.argmax(predictions, axis=1), average='macro')}")

# Print the recall of the model
print(
    f"Recall of the model: {recall_score(y_val, np.argmax(predictions, axis=1), average='macro')}")


# %%

print("Classification of Images using MobileNetV2")
print("Summary of the results:")
print(f"Number of Labelled Images: {num_of_images}")
print(f"Number of Training Images: {len(X_train)}")
print(f"Number of Validation Images: {len(X_val)}")
print(f"Number of Epochs: 20")
print(
    f"Accuracy of the model: {accuracy_score(y_val, np.argmax(predictions, axis=1))}")
print(
    f"F-measure of the model: {f1_score(y_val, np.argmax(predictions, axis=1), average='macro')}")
print(
    f"Confusion Matrix of the model:\n\n {confusion_matrix(y_val, np.argmax(predictions, axis=1))}")
print(
    f"Classification Report of the model:\n\n {classification_report(y_val, np.argmax(predictions, axis=1))}")
print(
    f"Precision of the model: {precision_score(y_val, np.argmax(predictions, axis=1), average='macro')}")
print(
    f"Recall of the model: {recall_score(y_val, np.argmax(predictions, axis=1), average='macro')}")

# Save the model
label_model.save('MobileNetV2.h5')

# Save the training and validation data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

# Save the predictions
np.save('predictions.npy', predictions)


# %% [markdown]
# Do the learning curve (Epochs is the x axis and accuracy is the y axis)
# Try to do data augmentation for classes with little images
#

# %%

# Load the model
model = load_model('MobileNetV2.h5')

# summary of the model
model.summary()

# Load the training and validation data
X_train = np.load('Training_data/X_train.npy')
X_val = np.load('Training_data/X_val.npy')
y_train = np.load('Training_data/y_train.npy')
y_val = np.load('Training_data/y_val.npy')


# %%
# plot the learning curve


model = load_model('MobileNetV2.h5')

# history of the training process
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val), verbose=0)


# evaluate the model on training and validation sets
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

# plot the learning curve based on the history of the training process
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# %%
