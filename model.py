'''
Created on 17.03.2018

@author: Hiep Truong
'''
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
# from statsmodels.sandbox.distributions.examples.matchdist import right_incorrect
    
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
def Plot_Images(images, number_img_per_row, title = None):
    """
        This function plot an arbitrary number of images
            input: 
                images: a numpy array of images
                title: a title for the plot
    """
    image_number = len(images)
    if image_number == 1:
        plt.title(title)
        if (images[0].ndim == 2) or (images[0].shape[2] == 1):
            plt.imshow(images[0], cmap='gray')
        else:
            plt.imshow(images[0])
    else:
        if image_number % number_img_per_row == 0:
            number_row = image_number / number_img_per_row
        else:
            number_row = image_number / number_img_per_row + 1
        fig, axs = plt.subplots(int(number_row),number_img_per_row, 
                            figsize=(16, 4 * image_number/number_img_per_row))
        if title!=None:
            fig.suptitle(title, fontsize=18)
        axs = axs.ravel()    
        for n in range(0,image_number):
            axs[n].axis('off')
            if images[n].ndim == 2:
                axs[n].imshow(images[n].squeeze(), cmap='gray')
            elif images[n].shape[2] == 1:
                axs[n].imshow(images[n].squeeze(), cmap='gray')
            else:
                axs[n].imshow(images[n])
    plt.show()

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

def correct_path_in_data(names):
    corrected_names = []
    for name in names:
        corrected_names.append(name.replace("IMG/", "./Data"))
    return corrected_names

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


File_name = './output/driving_log.csv'
 
center_image_names =  Read_Csv(File_name, 0, ',')
center_image_names = correct_path(center_image_names, r"C:\Users", "C:\\Users")
left_image_names = Read_Csv(File_name, 1, ',')
left_image_names = correct_path(left_image_names, r"C:\Users", "C:\\Users")
rigt_image_names = Read_Csv(File_name, 2, ',')
rigt_image_names = correct_path(rigt_image_names, r"C:\Users", "C:\\Users")
measurements = Read_Csv(File_name, 3, ',', dataType = 'float')
left_measurements = correct_measurement(measurements, +0.2)
right_measurements = correct_measurement(measurements, -0.2)
 
center_images = read_Img(center_image_names)
left_images = read_Img(left_image_names)
right_images = read_Img(rigt_image_names)
#  
flipped_center_images = flip_image_data(center_images)
flipped_measurements = flip_measurements(measurements)

File_name_reversed_laps = './2_Laps_reverse/driving_log.csv'
   
center_image_names_reversed_laps =  Read_Csv(File_name_reversed_laps, 0, ',')
center_image_names_reversed_laps = correct_path(center_image_names_reversed_laps,r"C:\Users", "C:\\Users")
left_image_names_reversed_laps = Read_Csv(File_name_reversed_laps, 1, ',')
left_image_names_reversed_laps = correct_path(left_image_names_reversed_laps,r"C:\Users", "C:\\Users")
rigt_image_names_reversed_laps = Read_Csv(File_name_reversed_laps, 2, ',')
rigt_image_names_reversed_laps = correct_path(rigt_image_names_reversed_laps, r"C:\Users", "C:\\Users")
measurements_reversed_laps = Read_Csv(File_name_reversed_laps, 3, ',', dataType = 'float')
left_measurements_reversed_laps = correct_measurement(measurements_reversed_laps, +0.2)
right_measurements_reversed_laps = correct_measurement(measurements_reversed_laps, -0.2)
   
center_images_reversed_laps = read_Img(center_image_names_reversed_laps)
left_images_reversed_laps = read_Img(left_image_names_reversed_laps)
right_images_reversed_laps = read_Img(rigt_image_names_reversed_laps)
   
flipped_center_images_reversed_laps = flip_image_data(center_images_reversed_laps)
flipped_measurements_reversed_laps = flip_measurements(measurements_reversed_laps)

File_name_data = './data/driving_log.csv'
 
center_image_names_data =  Read_Csv(File_name_data, 0, ',')
center_image_names_data = correct_path(center_image_names_data, r"IMG", "./data/IMG")
left_image_names_data = Read_Csv(File_name_data, 1, ',')
left_image_names_data = correct_path(left_image_names_data, r"IMG", "./data/IMG")
rigt_image_names_data = Read_Csv(File_name_data, 2, ',')
rigt_image_names_data = correct_path(rigt_image_names_data, r"IMG", "./data/IMG")
measurements_data = Read_Csv(File_name_data, 3, ',', dataType = 'float')
 
center_images_data = read_Img(center_image_names_data)
# left_images_data = read_Img(left_image_names_data)
# right_images_data = read_Img(rigt_image_names_data)
 
# flipped_center_images_data = flip_image_data(center_images_data)
# plt.imshow(flipped_center_images_data[0])
# plt.show()
# flipped_measurements_data = flip_measurements(measurements_data)

# X_train = np.array(center_images_data + center_images)
# y_train = np.array(measurements_data + measurements)

X_train = np.array(center_images+ left_images + right_images 
                   + center_images_reversed_laps + left_images_reversed_laps + right_images_reversed_laps)
y_train = np.array(measurements+ left_measurements + right_measurements
                   + measurements_reversed_laps + left_measurements_reversed_laps + right_measurements_reversed_laps)
print(X_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(90,320,3)))
# model.add(Cropping2D(cropping=((50,20), (0,0))))


# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(16,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))


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

model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')

