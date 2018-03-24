'''
Created on 17.03.2018

@author: Hiep Truong
'''
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
    
def Read_Csv(file_name, column_idx , delimiter, dataType = None):
    """
        This function reads a column from a csv file
            input: 
                file_name: name of the csv file
                column_idx: the index of the column to be read
                delimiter: column delimiter symbol
            return: a list of data
    """
    data = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for num,row in enumerate(reader):
            if num >= 1:
                if dataType == 'float':
                    data.append(float(row[column_idx]))
                else:
                    data.append(row[column_idx])
    return data

def read_Img(file_names):
    images = []
    for name in file_names:
        img = cv2.imread(name)
        images.append(img[50:140,:])
    return images

def correct_path(names, to_be_replaced, to_replace ):
    corrected_names = []
    for name in names:
        corrected_names.append(name.replace(to_be_replaced, to_replace))
    return corrected_names

def correct_measurement(measurements, correct_value = 0.0):
    corrected = []
    for measurement in measurements:
        corrected.append(measurement + correct_value)
    return corrected

def flip_image_data(images):
    flipped_images = []
    for image in images:
        flipped_images.append(cv2.flip(image, 1))
    return flipped_images

def flip_measurements(measurments):
    flipped_measurements = []
    for measurement in measurments:
        flipped_measurements.append(measurement * -1.0)
    return flipped_measurements

def get_data(File_name, to_be_replaced, to_replace):
    center_image_names =  Read_Csv(File_name, 0, ',')
    center_image_names = correct_path(center_image_names, to_be_replaced, to_replace)
    left_image_names = Read_Csv(File_name, 1, ',')
    left_image_names = correct_path(left_image_names, to_be_replaced, to_replace)
    rigt_image_names = Read_Csv(File_name, 2, ',')
    rigt_image_names = correct_path(rigt_image_names, to_be_replaced, to_replace)
    center_images = read_Img(center_image_names)
    left_images = read_Img(left_image_names)
    right_images = read_Img(rigt_image_names)
    measurements = Read_Csv(File_name, 3, ',', dataType = 'float')
    left_measurements = correct_measurement(measurements, +0.2)
    right_measurements = correct_measurement(measurements, -0.2)
    
    return center_images, left_images, right_images, measurements, left_measurements, right_measurements
    
File_name = './output/driving_log.csv'
center_images, left_images, right_images, measurements, left_measurements, right_measurements = get_data(File_name, r"C:\Users", "C:\\Users")

File_name_reversed_laps = './2_Laps_reverse/driving_log.csv'
center_images_reversed_laps, left_images_reversed_laps, right_images_reversed_laps, measurements_reversed_laps, left_measurements_reversed_laps, right_measurements_reversed_laps = get_data(File_name_reversed_laps, r"C:\Users", "C:\\Users")

X_train = np.array(center_images+ left_images + right_images 
                   + center_images_reversed_laps + left_images_reversed_laps + right_images_reversed_laps)
y_train = np.array(measurements+ left_measurements + right_measurements
                   + measurements_reversed_laps + left_measurements_reversed_laps + right_measurements_reversed_laps)

print(X_train.shape)
print(y_train.shape)

def Model_1():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(90,320,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def Model_2():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(90,320,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = Model_2()
model.compile(loss = 'mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
model.save('model.h5')

