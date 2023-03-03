import csv
import numpy as np

def parse_data(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        first_line = True
        temp_images, temp_labels = [], []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:]
                img_as_array = np.array_split(image_data, 28)
                temp_images.append(img_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
        return images, labels