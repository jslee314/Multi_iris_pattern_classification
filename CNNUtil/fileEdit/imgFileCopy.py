import os
import shutil
from CNNUtil.fileEdit.utility import Utility as Util


srcDir = 'D:/이재선/'
dstDir = 'D:/이재선/mask/'


Util.make_dir(dstDir)

dirNames = os.listdir(srcDir)
for dirName in dirNames:
    fileNames = os.listdir(srcDir + dirName)
    for fileName in fileNames:
        if fileName[:5] == 'iris_':
            print(srcDir + dirName)

            print(fileName[:-4])
            shutil.copy(srcDir + dirName + '/' + fileName, dstDir +'ab_'+ fileName[:-4] + '_mask.PNG')



srcDir = 'D:/이재선/mask/'
dstDir = 'D:/이재선/스트레스링/'


Util.make_dir(dstDir)

dirNames = os.listdir(srcDir)
for dirName in dirNames:
    fileNames = os.listdir(srcDir + dirName)
    for fileName in fileNames:
        if fileName[:5] == 'st_iris_':
            print(srcDir + dirName)

            print(fileName[:-4])
            shutil.copy(srcDir + dirName + '/' + fileName, dstDir +'ab_'+ fileName[:-4] + '_mask.PNG')
