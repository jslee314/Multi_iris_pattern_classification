import os
import cv2
import re

class Utility:

    def make_dir(name):
        if not os.path.isdir(name):
            os.makedirs(name)
            print(name, "폴더가 생성되었습니다.")
        # else:
        #     print("해당 폴더가 이미 존재합니다.")

    def put_text_in_image(image, text):
        # show the image
        cv2.putText(image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 10, 50), 3)
        return image

    # 파일명 변경함수
    def convert_file_name(srcDir):
       # srcDir = 'D:/Data/iris_pattern/Original/lacuna'

        dirNames = os.listdir(srcDir)
        for dirName in dirNames:
            fileNames = os.listdir(srcDir+'/'+dirName)
            for fileName in fileNames:
                filePath = srcDir + '/' + dirName + '/' + fileName
                print("origin  "+filePath)

                # case1 한글 지우기
                new_filePath = re.compile('[가-힣]+').sub('', filePath)

                # case2 뒤에 4글자 지우기
                #new_filePath = filePath[:-4]

                print(new_filePath)
                os.rename(filePath, new_filePath)