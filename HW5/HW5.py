#************************************************************************************
# Shafiqul Alam Khan
# ML – HW#5
# Filename: HW5.py
# Due: Nov. 1, 2023
#
# Objective:
# • Develop your own CNN model to classify all classes.
# • Provide the training and test confusion matrices.
# • Provide the test accuracy, precision, recall, and F1-scores to a text file.
# • Provide the Loss and Accuracy curves for training and validation (you can use a single plot for these four curves)
# • Expected results: High 90’s for training, validation, and testing without overfitting/underfitting.
#*************************************************************************************

s = open("Output_HW5.txt", "w")
s.write('>> Importing Packages...\n')
# Importing all required libraries
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
s.write('\n\t\t\t\t....DONE!\n')

# Epochs
ep = 50
# Batch size
bsize = 10

####################################################
#
# Data Preprocessing
#
####################################################

# s.write('>> Making train, validation & test directories...\n')

# #Make train, validation & test directories
# os.mkdir('train')
# os.mkdir('validation')
# os.mkdir('test')
# s.write('\n\t\t\t\t....DONE!\n')

# s.write('>> Renaming files...\n')

# # Rename all the files of Concrete_Crack_Images_for_Classification/Negative directory
# for rn in range(1, 20001):
#     old_nm = str(rn)+'.jpg'
#     new_nm = 'negative.'+str(rn)+'.jpg'
#     old_file = os.path.join('Concrete_Crack_Images_for_Classification/Negative', old_nm)
#     new_file = os.path.join('Concrete_Crack_Images_for_Classification/Negative', new_nm)
#     os.rename(old_file, new_file)
    
# # Rename all the files of Concrete_Crack_Images_for_Classification/Positive directory
# for rn in range(1, 20001):
#     old_nm = str(rn)+'.jpg'
#     new_nm = 'positive.'+str(rn)+'.jpg'
#     old_file = os.path.join('Concrete_Crack_Images_for_Classification/Positive', old_nm)
#     new_file = os.path.join('Concrete_Crack_Images_for_Classification/Positive', new_nm)
#     os.rename(old_file, new_file)

# s.write('\n\t\t\t\t....DONE!\n')

# s.write('>> Moving files to train, validation & test directories...\n')
# # Move files to test folder
# for rn in range(1, 2001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Negative/negative.'+str(rn)+'.jpg'
#     dst_dir = 'test'
#     shutil.copy(src_file, dst_dir)

# for rn in range(1, 2001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Positive/positive.'+str(rn)+'.jpg'
#     dst_dir = 'test'
#     shutil.copy(src_file, dst_dir)

# # Move files to validation folder
# for rn in range(2001, 3001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Negative/negative.'+str(rn)+'.jpg'
#     dst_dir = 'validation'
#     shutil.copy(src_file, dst_dir)

# for rn in range(2001, 3001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Positive/positive.'+str(rn)+'.jpg'
#     dst_dir = 'validation'
#     shutil.copy(src_file, dst_dir)

# # Move files to train folder
# for rn in range(3001, 20001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Negative/negative.'+str(rn)+'.jpg'
#     dst_dir = 'train'
#     shutil.copy(src_file, dst_dir)

# for rn in range(3001, 20001):
#     src_file = 'Concrete_Crack_Images_for_Classification/Positive/positive.'+str(rn)+'.jpg'
#     dst_dir = 'train'
#     shutil.copy(src_file, dst_dir)

# s.write('\n\t\t\t\t....DONE!\n')

s.write('>> Preprocessing data...\n')
# Define train, test & validation directory
train_dir = 'train'
test_dir = 'test'
validation_dir = 'validation'

# Use ImageDataGenerator to create variations
train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, 
                                   rescale=1./255)

validation_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, 
                                   rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(227, 227), 
                                                    batch_size=bsize, class_mode='binary', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(227, 227), 
                                                    batch_size=bsize, class_mode='binary', shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(227,227), 
                                                  batch_size=1, class_mode='binary', shuffle=False)

s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# CNN
#
####################################################

s.write('>> Creating the CNN...\n')
CNN = models.Sequential()
CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(227,227,3)))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Dropout(0.5))
CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(256, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Dropout(0.5))
CNN.add(layers.Conv2D(512, (3, 3), activation='relu'))
CNN.add(layers.Flatten())
CNN.add(layers.Dense(512, activation='relu'))
CNN.add(layers.Dense(1, activation='sigmoid'))
CNN.summary()

s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# Optimize / Train & Save
#
####################################################
s.write('>> Optimize / Train & Save Model...\n')

CNN.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4), metrics=['acc'])

history = CNN.fit(train_generator, steps_per_epoch=100,
                  epochs=ep, validation_data=validation_generator, validation_steps=50)


CNN.save('Concrete_Crack.keras')

s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# Plot result
#
####################################################

s.write('>> Generate plots...\n')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Output_HW_5_Training_and_validation_accuracy.png')
plt.clf()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Output_HW_5_Training_and_validation_loss.png')
plt.clf()

s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# Confusion matrices
#
####################################################
s.write('\n>> Plotting the confusion matrices... \n')

# Print the Target names
target_names = []
for key in train_generator.class_indices:
    target_names.append(key)
    
# Training Confution Matrix
Y_pred_train = CNN.predict(train_generator)
y_pred_train = np.where(Y_pred_train>0.5, 1, 0)
y_train = train_generator.classes
cm_train = confusion_matrix(y_train, y_pred_train)

# Plot & save training confusion matrix
sns.heatmap(cm_train,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = target_names,
            yticklabels = target_names)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Training confusion matrix', fontsize = 14)
plt.savefig('Output_HW_5_Training_Confusion_Matrix.png')
plt.clf()

# Test Confution Matrix
Y_pred_test = CNN.predict(test_generator)
y_pred_test = np.where(Y_pred_test>0.5, 1, 0)
y_test = test_generator.classes
cm_test = confusion_matrix(y_test, y_pred_test)

# Plot & save test confusion matrix
sns.heatmap(cm_test,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = target_names,
            yticklabels = target_names)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Test confusion matrix', fontsize = 14)
plt.savefig('Output_HW_5_Test_Confusion_Matrix.png')
plt.clf()

s.write('\n\t\t\t\t....DONE!\n')

####################################################
#
# Accuracy, Precision, Recall, and F1-scores
#
###################################################
s.write('\n>> Print Accuracy, Precision, Recall, and F1-scores... \n')

s.write('\n> Test Accuracy: %0.4f' %(accuracy_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test Precision: %0.4f' %(precision_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test Recall: %0.4f' %(recall_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test F1-scores: %0.4f' %(f1_score(y_test, y_pred_test)) + '\n')

s.write('\n\t\t\t\t....DONE!\n')

s.write('\n******************PROGRAM is DONE *******************')
s.close()

