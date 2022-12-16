import cv2
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#HÀM KIỂM TRA ẢNH CÓ LỖI HAY KHÔNG
def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            io.imread(os.path.join(img_file))
        return False
    except Exception as e:
        print(e)
        return True

def read_img_data(path,label, size):
    X = []
    y = []
    files = os.listdir(path)
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray=True)
            img = resize(img, size)
            img_vector = img.flatten()
            X.append(img_vector)
            y.append(label)
    X = np.array(X)
    return X,y

#hàm chuyển ảnh màn hình thành vector 1024
def convert_D_2_vector(path,label,size):
    labels = []
    img_data = []
    images = os.listdir(path)
    for img_file in images:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img_grey = io.imread(os.path.join(path,img_file), as_grey = True)
            img_vector = resize(img_grey,size).flatten()
            img_data.append(label)
    return img_data, labels

#hàm xây dựng cơ sở dữ liệu ảnh mô hình
def build_img_data():
    cat_path = read_img_data('C:/Users/Ngoc Ha/Desktop/hoc may/PetImages/Cat', 'Cat', (32,32))
    dog_path = read_img_data('C:/Users/Ngoc Ha/Desktop/hoc may/PetImages/Dog', 'Dog', (32,32))
    images,labels = convert_D_2_vector(cat_path, 'Cat', (32,32))
    img_dogs, label_dogs = convert_D_2_vector(dog_path, 'Dog', (32,32))
    images.extend(img_dogs)
    labels.extend(label_dogs)
    X = np.array(images)
    y = LabelBinarizer().fit_transform(labels)
    return X,y

def train_test(D_dog, D_cat):
    D_dog, D_cat = D[:, :-1], D[:, -1]
    X_train, X_test, y_train, y_t