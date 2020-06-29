from .constants import *
from .showtrain import hist_saved, confusion_matrix_saved
from .saver import ModelSaver
from sklearn.metrics import classification_report, confusion_matrix


def makeoutput(x_val, y_val, model, hist, classes):
    print('[OUTPUT 1] model.evaluate)')
    loss_and_metrics = model.evaluate(x_val, y_val, batch_size=FLG.BATCH_SIZE)
    print('loss_and_metrics : ' + str(loss_and_metrics))

    print('[OUTPUT 2] 모델 학습 과정 그래프 저장: png')
    hist_saved(hist)

    print('[OUTPUT 3] 모델 저장하기: h5, ckpt')
    modelSaver = ModelSaver(model)
    modelSaver.h5saved()
    modelSaver.ckptsaved()
    modelSaver.pbsaved()


     # 저장한 모델 불러와서 predict 데이터 set 에 대하여 테스트 진행
    print('[OUTPUT 4] model.predict:  classification_report  & confusion_matrix')
    predictions = model.predict(x_val, batch_size=FLG.BATCH_SIZE)

    f = open(FLG.CONFUSION_MX, 'w')

    print('## classification_report')
    target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    classfi_report = classification_report(y_val.argmax(axis=1),
                                           predictions.argmax(axis=1),
                                           target_names=target_names)
    print(classfi_report)
    f.write(str(classfi_report))

    print('## confusion_matrix')
    confu_mx = confusion_matrix(y_val.argmax(axis=1),
                                predictions.argmax(axis=1))

    print(confu_mx)
    f.write(str(confu_mx))

    f.close()

    print('## confusion_matrix plot')
    confusion_matrix_saved(confu_mx, classes)












