import dlib
import face_recognition
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('img/aobama.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 
# detector = dlib.get_frontal_face_detector() 
# dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
# for index, face in enumerate(dets):
#     # 人脸上下左右的坐标位置
#     left = face.left()
#     top = face.top()
#     right = face.right()
#     bottom = face.bottom()
#     
#     print((left, top, right, bottom))
#     img[top][left:right] = [255,0,0]
#     img[bottom][left:right] = [255,0,0]
# #     img[top:bottom][left] = [255,0,0]
# #     img[top:bottom][right] = [255,0,0]
#     
#     plt.imshow(img)
#     plt.show()


image = face_recognition.load_image_file('img/obama.jpg')
# 人脸区域
face_locations = face_recognition.face_locations(image)
# 人脸64个特征坐标
face_landmarks_list = face_recognition.face_landmarks(image)
# 人脸特征矩阵（n, 128维）
known_encodings = face_recognition.face_encodings(image)
print(known_encodings)

face_landmarks = face_landmarks_list[0]
point2color = {
    "chin": [0,255,255],
    "left_eyebrow": [255,255,0],
    "right_eyebrow": [255,255,0],
    "nose_bridge": [255,0,255],
    "nose_tip": [0,255,0],
    "left_eye": [0,0,255],
    "right_eye": [0,0,255],
    "top_lip": [255,255,255],
    "bottom_lip": [255,0,0],
}
for (key, points) in face_landmarks.items():
    for point in points:
        image[point[1]][point[0]] = point2color[key]

# plt.imshow(image)
# plt.show()



unknown_image = face_recognition.load_image_file('img/obama2.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)
print(unknown_encoding)
results = face_recognition.compare_faces(known_encodings, unknown_encoding[0], 0.6)
face_distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])
print(results, face_distances)



