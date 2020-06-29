import os
import cv2
import numpy as np
import random

from CNNUtil.fileEdit.utility import Utility as Util


# 파일 이름에 한글이 들어가있으면 지우기
#Util.convert_file_name( 'D:/Data/iris_pattern/Original')


'[STEP] Original image dataset_2 을 각 카테버리별 리스트에 저장'

srcDir = 'D:/Data/iris_pattern/Original_2/train/'
original_shape = [120, 120, 3]

lacunas = []        # 열공
defects = []        # 결절
spokes = []         # 바큇살
spots = []            # 색소반점
normals = []         # 정상

filePaths = os.listdir(srcDir)

for filePath in filePaths:
    imagePaths = os.listdir(srcDir+filePath)
    for imagePath in imagePaths:
        #print(srcDir+filePath + '/' + imagePath)
        image = cv2.imread(srcDir+filePath + '/' + imagePath)
        # cv2.imshow('before', image)
        # cv2.waitKey(1000)
        image = cv2.resize(image, (original_shape[1], original_shape[0]))
        # cv2.imshow('after', image)
        # cv2.waitKey(1500)
        if filePath =='lacuna':
            lacunas.append(image)
        elif filePath =='defect':
            defects.append(image)
        elif filePath =='spoke':
            spokes.append(image)
        elif filePath =='spot':
            spots.append(image)
        elif filePath == 'normal':
            normals.append(image)
        else:
            print("포함된 카테고리 없음")

nullPadding = np.zeros(original_shape)
nullPaddings = [nullPadding, nullPadding]


' [STEP] 데이터에 들어갈 4개의 이미지를 각 카테고리 별 랜덤으로 뽑는다.'

# 멀티라벨 데이터 만들 이미지 패턴 6가지(열공, 결절, 바큇살, 색소반점, 정상, null값)에 대한 딕셔너리생성  --   key(파일이름을위해), value(해당되는 이미지리스트들)
#classDict = {'lacuna': lacunas, 'defect': defects,  'spoke' : spokes, 'spot': spots, 'normal': normals, 'padding': nullPaddings}
classDict = {'lacuna': lacunas, 'defect': defects, 'spot': spots,  'spoke': spokes}

def make_multi_img(img_num):

    for num in range(img_num*1):
        make_multi_img_random(num, 4)
    for num in range(img_num*4):
        make_multi_img_random(num, 3)
    for num in range(img_num*6):
        make_multi_img_random(num, 2)
    for num in range(img_num*4):
        make_multi_img_random(num, 1)
    for num in range(img_num):
        make_multi_img_random(num, 0)

def make_multi_img_random(num, padding_num):

    # 6개중 4가지 이미지패턴 이름을 랜덤으로 추출 후 사전순으로 정렬
    selectedClassses = random.sample(classDict.keys(), 4 - padding_num)
    sortedClassses = sorted(selectedClassses)

    # 4개의 패턴의 키값 리스트를 통해 해당패턴마다 랜덤으로 하나씩 이미지를 추출하여  imgSet  리스트에 저장(4개임)
    imgFileName = ''
    imgSet = []
    for i in range(len(sortedClassses)):
        key = sortedClassses[i]
        val = classDict[key]
        random_num = random.randrange(0, len(val) - 1, 1)
        imgSet.append(val[random_num])
        # if key !='padding':
        #     imgFileName = imgFileName + key + '_'

    x = 'defect' if 'defect' in selectedClassses else 'normal'
    imgFileName = imgFileName + '_' + x

    x = 'lacuna' if 'lacuna' in selectedClassses else 'normal'
    imgFileName = imgFileName + '_' + x

    x = 'spoke' if 'spoke' in selectedClassses else 'normal'
    imgFileName = imgFileName + '_' + x

    x = 'spot' if 'spot' in selectedClassses else 'normal'
    imgFileName = imgFileName + '_' + x



    for i in range(padding_num):
        # imgSet.append(nullPadding)
        r_num = random.randrange(0, len(normals) - 1, 1)
        imgSet.append(normals[r_num])


    '[STEP] 선택된 4개의 이미지를 순서를 랜덤하게 한 후  2x2 1개의 이미지로 만든다'
    dstDir = 'D:/Data/iris_pattern/Multi_output2_test40_train160/train/'
    random.shuffle(imgSet)
    row1 = np.concatenate((imgSet[0], imgSet[1]), axis=1)
    row2 = np.concatenate((imgSet[2], imgSet[3]), axis=1)
    concat = np.concatenate((row1, row2), axis=0)
    imgFileName = imgFileName[1:]
    Util.make_dir(dstDir +imgFileName+'/')
    cv2.imwrite(dstDir +imgFileName+'/' + imgFileName + str(num) + ".png", concat)

make_multi_img(160)

