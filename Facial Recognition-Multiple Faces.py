from deepface import DeepFace
import os
import time
import cv2
start=time.time()
curr_dir = os.getcwd()

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

# dfs = DeepFace.find(
#   img_path = "Pic_3.jpg",
#   db_path = "Database/Person 2",
# )



def check_folder(picture):
    count_files = 0
    count_success = 0
    count_number_people = 0
    success_count = 0
    pic = picture

    for p in os.listdir('Database'):
        person = p.split("-")[0]
        p_name = p.split("-")[1]
        count_number_people += 1
        for person_pictures in os.listdir(os.path.join('Database', p)):
            count_files += 1
            # print(person_pictures, person)
            data = DeepFace.verify(pic, 'Database/' + str(person) + "-" + p_name + '/Pic_' + str(count_files) + '.jpg', model_name = models[2])
            # print(data)
            if data["distance"]<0.4:
                count_success +=1
                # data = DeepFace.verify(pic, 'Database/' + str(person) + '/Pic_' + str(count_files) + '.jpg')
                print ("Success")
                # print (data)
        print(count_success, count_files)
        if count_success>=0.4*count_files and count_files<10:
            imgSaveDir = os.path.join('Database/', p, 'Pic_' + str(count_files+1) + '.jpg')
            print(f"Hello {p_name}")
            cv2.imwrite(imgSaveDir , pic)
            success_count += 1
            
        count_files=0
        count_success=0
        
    if success_count==0:
        os.mkdir('Database/Person ' + str(count_number_people+1))
        save_image_to_new_directory(pic, "Pic_1.jpg", 'Database/Person ' + str(count_number_people+1))
 

def save_image_to_new_directory(image, image_name, directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    image_path = os.path.join(directory_name, image_name)
    success = cv2.imwrite(image_path, image)

    
img = cv2.imread('Duo Pics Check/Pic_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces_rect = face_classifier.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)


print(f'Number of faces found is: {len(faces_rect)}')


for (x,y,w,h) in faces_rect:
    
    image = img[y-100:y+h+100, x-100:x+w+100]
    cv2.waitKey(1000)
    
    
    check_folder(image)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    # cv.imshow('Image',image)
    
    # cv.waitKey(5000)
    # Draws a bounding rectangle using coordinates from haar classifier

cv2.imshow('detected face', img)
end=time.time()
print(end-start)
cv2.waitKey(0)