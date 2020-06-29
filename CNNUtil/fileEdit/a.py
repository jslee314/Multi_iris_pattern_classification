'''
# a, b, c, 리스트에 공통으로 들어있는 요소 뽑기.
a = [1, 2, 3, 4, 5, 6, 10]
b = [2, 4, 6, 8, 10]
c = [0, 5, 10, 15]

result = list(set(a) & set(b) & set(c))
result_b = [x for x in a if x in [y for y in b if y in c]]

print(result)
print(result_b)
'''

import os
import shutil

# src_ab = os.listdir('D:/test/ab')
# src_ch = os.listdir('D:/test/ch')
# src_st = os.listdir('D:/test/st')
#
# # ab, ch, st  디렉토리에 있는 파일 리스트들 중 3곳에 모두 공통으로 있는 파일 리스트만 추출
# common_set = list(set(src_ab) & set(src_ch) & set(src_st))

#
# srcDir = 'D:/Data/cls/Single-label'
#
# dirNames = os.listdir(srcDir)
# for dirName in dirNames:
#     fileNames = os.listdir(srcDir+'/'+dirName)
#     for fileName in fileNames:
#         imgNames = os.listdir(srcDir + '/' + dirName + '/' + fileName)
#         for imgName in imgNames:
#             # filePath = srcDir + '/' + dirName + '/' + fileName + '/' + imgName
#             # new_filePath = filePath[:-4]
#             # #print(filePath)
#             # #print(new_filePath)
#             # os.rename(filePath, new_filePath)
#
#             if imgName not in common_set:
#                 filePath = srcDir + '/' + dirName + '/' + fileName + '/' + imgName
#                 print(filePath)
#                 os.remove(filePath)
'''
삭제한 데이터들
D:/Data/cls/ab/0/20191024_8b3a9528-beac-476d-b3e8-2827518e77d2.png.png
D:/Data/cls/ab/1/20191008_df4b6d13-0eb1-42dd-8ef0-784c07595bd8.png.png
D:/Data/cls/ch/0/20190826_3c9981c2-5d97-4f5b-8493-a839c866b1b0.png.png
D:/Data/cls/ch/0/20190826_4401daaf-d12d-472a-b8d5-663f4c5e126f.png.png
D:/Data/cls/ch/0/20191009_7114f29a-7ecd-4ee2-911a-e2aac3c7822f.png.png
D:/Data/cls/ch/0/20191024_8b3a9528-beac-476d-b3e8-2827518e77d2.png.png
D:/Data/cls/ch/1/20191008_df4b6d13-0eb1-42dd-8ef0-784c07595bd8.png.png
D:/Data/cls/st/0/20190826_4401daaf-d12d-472a-b8d5-663f4c5e126f.png.png
D:/Data/cls/st/1/20190826_3c9981c2-5d97-4f5b-8493-a839c866b1b0.png.png
'''

ab_0 = os.listdir('D:/Data/cls/Single-label/ab/0')
ab_1 = os.listdir('D:/Data/cls/Single-label/ab/1')
ch_0 = os.listdir('D:/Data/cls/Single-label/ch/0')
ch_1 = os.listdir('D:/Data/cls/Single-label/ch/1')
st_0 = os.listdir('D:/Data/cls/Single-label/st/0')
st_1 = os.listdir('D:/Data/cls/Single-label/st/1')


nm = list(set(ab_0) & set(ch_0) & set(st_0))
ab = list(set(ab_1) & set(ch_0) & set(st_0))
ch = list(set(ab_0) & set(ch_1) & set(st_0))
st = list(set(ab_0) & set(ch_0) & set(st_1))
ab_ch = list(set(ab_1) & set(ch_1) & set(st_0))
ab_st = list(set(ab_1) & set(ch_0) & set(st_1))
ch_st = list(set(ab_0) & set(ch_1) & set(st_1))
ab_ch_st = list(set(ab_1) &set(ch_1) & set(st_1))

print('각 집합별로 들어가있는 리스트 수 체크')
print(str(len(nm)) +' / ' +str(len(ab)) +' / ' + str(len(ch)) +' / ' + str(len(st)) +' / ' + str(len(ab_ch)) +' / ' + str(len(ab_st)) +' / ' + str(len(ch_st)) +' / ' + str(len(ab_ch_st)))
print('total : ' + str(len(nm) + len(ab) + len(ch) + len(st) + len(ab_ch) + len(ab_st) + len(ch_st)+len(ab_ch_st)))


SrcFileDir = 'D:/Data/cls/Original/'
DstFileDir = 'D:/Data/cls/Multi-label/'

SrcFileNames = os.listdir(SrcFileDir)
for SrcfileName in SrcFileNames:
    dir = ''
    if SrcfileName in nm:
        dir = 'nm/'
    elif SrcfileName in ab:
        dir = 'ab/'
    elif SrcfileName in ch:
        dir = 'ch/'
    elif SrcfileName in st:
        dir = 'st/'
    elif SrcfileName in ab_ch:
        dir = 'ab_ch/'
    elif SrcfileName in ab_st:
        dir = 'ab_st/'
    elif SrcfileName in ch_st:
        dir = 'ch_st/'
    elif SrcfileName in ab_ch_st:
        dir = 'ab_ch_st/'
    else:
        print("조건에 속한 집합이 없음")
    shutil.copy(SrcFileDir + SrcfileName, DstFileDir + dir + SrcfileName)







