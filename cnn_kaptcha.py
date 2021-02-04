import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 图片去噪
def filterNoise(originImg):
    newImg = np.zeros([originImg.shape[0], originImg.shape[1], ])
    for rowIndex, row in enumerate(originImg):
        for colIndex, col in enumerate(row):
            if col[0] == col[1] and col[1] == col[2]:
                newImg[rowIndex][colIndex] = 0.
            else:
                gray = np.mean(col)
                if gray > 0.5:
                    gray = 1
                else:
                    gray = 0
                newImg[rowIndex][colIndex] = np.mean(gray)
    return newImg


# 获得训练图片
test_images = []
baseDir = "C:/Users/wuxb/Desktop/kaptcha_cnn/test/"
pathDir =  os.listdir(baseDir)
for fileDir in pathDir:
    img = mpimg.imread(baseDir + fileDir)
    img = filterNoise(img)
#     originImg = originImg / 255.0
    test_images.append(img)
    print(img)
    break
test_images = np.array(test_images)


plt.imshow(test_images[0], cmap='gray')
plt.show()

print(test_images)

# with open("C:/Users/wuxb/Desktop/kaptcha_cnn/test_answer.json","r") as f:
#     answerData = json.load(f)
# 
# print(answerData)
