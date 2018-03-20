'''
Created on 17.03.2018

@author: Hiep Truong
'''
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
    
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
        images.append(cv2.imread(name))
    return images

def correct_path(names):
    corrected_names = []
    for name in names:
        corrected_names.append(name.replace(r"C:\Users\H", "C:\\Users\H"))
    return corrected_names

def flip_image_data(images):
    flipped_images = []
    for image in images:
        flipped_images.append(cv2.flip(image, 0))
    return flipped_images

def flip_measurements(measurments):
    flipped_measurements = []
    for measurement in measurments:
        flipped_measurements.append(-measurement)
    return flipped_measurements


File_name = './output/driving_log.csv'

center_image_names =  Read_Csv(File_name, 0, ',')
center_image_names = correct_path(center_image_names)
left_image_names = Read_Csv(File_name, 1, ',')
left_image_names = correct_path(left_image_names)
rigt_image_names = Read_Csv(File_name, 2, ',')
rigt_image_names = correct_path(rigt_image_names)
measurements = Read_Csv(File_name, 3, ',', dataType = 'float')


center_images = read_Img(center_image_names)
left_images = read_Img(left_image_names)
right_images = read_Img(rigt_image_names)

flipped_center_images = flip_image_data(center_images)
flipped_measurements = flip_measurements(measurements)

File_name_reversed_laps = './2_Laps_reverse/driving_log.csv'
 
center_image_names_reversed_laps =  Read_Csv(File_name_reversed_laps, 0, ',')
center_image_names_reversed_laps = correct_path(center_image_names_reversed_laps)
left_image_names_reversed_laps = Read_Csv(File_name_reversed_laps, 1, ',')
left_image_names_reversed_laps = correct_path(left_image_names_reversed_laps)
rigt_image_names_reversed_laps = Read_Csv(File_name_reversed_laps, 2, ',')
rigt_image_names_reversed_laps = correct_path(rigt_image_names_reversed_laps)
measurements_reversed_laps = Read_Csv(File_name_reversed_laps, 3, ',', dataType = 'float')
 
 
center_images_reversed_laps = read_Img(center_image_names_reversed_laps)
left_images_reversed_laps = read_Img(left_image_names_reversed_laps)
right_images_reversed_laps = read_Img(rigt_image_names_reversed_laps)
 
flipped_center_images_reversed_laps = flip_image_data(center_images_reversed_laps)
flipped_measurements_reversed_laps = flip_measurements(measurements_reversed_laps)





# X_train = np.array(center_images + flipped_center_images + center_images_reversed_laps + flipped_center_images_reversed_laps)
# y_train = np.array(measurements + flipped_measurements + measurements_reversed_laps + flipped_measurements_reversed_laps)

X_train = np.array(center_images+flipped_center_images)
y_train = np.array(measurements+flipped_measurements)
print(X_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=10)

model.save('model.h5')

