import cv2
import os
import numpy as np

def get_path_list(root_path):
    # To get a list of path directories from root path
    actors_name = os.listdir(root_path)
    return actors_name

def get_class_names(root_path, train_names):
    # To get a list of train image and a list of image class
    train_image_list = []
    image_classes_list = []
    for index, train_name in enumerate(train_names):
        image_path_list = os.listdir(root_path+'/'+train_name)
        for image_path in image_path_list:
            train_image_list.append(root_path+'/'+train_name+'/'+image_path)
            image_classes_list.append(index)

    return train_image_list, image_classes_list
    

def detect_faces_and_filter(image_list, image_classes_list=None):
    # To detect a face from given image list and filter it if the face on
    #    the given image is more or less than one
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_list = []
    face_rectangle = []
    class_list = []
    for i in range(len(image_list)):
        img_bgr = cv2.imread(image_list[i])
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors = 5)
        if(len(detected_faces) < 1):
            continue
        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
            cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,0,255), 1)
            face_list.append(face_img)
            if(image_classes_list == None):
                continue
            else:
                class_list.append(image_classes_list[i])
    
    return face_list, face_rectangle, class_list
    

def train(train_face_grays, image_classes_list):
    # Create and train recognizer object
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))
    return face_recognizer

def get_test_images_data(test_root_path):
    # Load a list of test images from given path list
    test_image_list = []
    for test_image_path in os.listdir(test_root_path):
        full_image_path = test_root_path+'/'+test_image_path
        test_image_list.append(full_image_path)
    return test_image_list


def predict(recognizer, test_faces_gray):
    # predict the test image with recognizer
    result = []
    for i in range(len(test_faces_gray)):
        res, loss = recognizer.predict(test_faces_gray[i])
        result.append(res)
    return result

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    # Draw prediction results on the given test images
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    res_img = []
    for i in range(len(test_image_list)):
        img_bgr = cv2.imread(test_image_list[i])
        img_bgr = cv2.resize(img_bgr, (200, 200))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors = 5)
        if(len(detected_faces) < 1):
            continue
        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
            cv2.rectangle(img_bgr, (x,y), (x+w, y+h),(0, 255 , 0), 1)
            text = train_names[predict_results[i]]
            cv2.putText(img_bgr, text, (x, y-2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            res_img.append(img_bgr)
    return res_img


    
def combine_and_show_result(image_list):
    # To show the final image that already combine into one image
    #    Before the image combined, it must be resize with
    #    width and height : 200px
    for i in range(len(image_list)):
        image_list[i] = cv2.resize(image_list[i], (200, 200))
    combine_all_images = cv2.hconcat([image for image in image_list])
    cv2.imshow("Final Result", combine_all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    train_root_path = 'dataset/train'
    
    # Get Training Data
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)    

    
    test_root_path = 'dataset/test'
    
    # Get Testing Data
    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    combine_and_show_result(predicted_test_image_list)    