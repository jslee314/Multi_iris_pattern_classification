from .constants import *
from .showtrain import hist_saved, confusion_matrix_saved
from .saver import ModelSaver
from sklearn.metrics import classification_report, confusion_matrix
import os

def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)
        print(name, "폴더가 생성되었습니다.")
    else:
        print("해당 폴더가 이미 존재합니다.")

def saved_model_and_graph(model, hist, output_list):
    print('[결과저장 1]   모델 학습 과정 그래프 저장: png')
    hist_saved(hist, output_list)

    print('[결과저장 2]   모델 저장하기: h5, ckpt')
    modelSaver = ModelSaver(model)
    modelSaver.h5saved()

    # modelSaver.save_tfLite()
    # modelSaver.ckptsaved()
    # modelSaver.pbsaved()


def saved_matrix(x_val, y_val, model, classes):

    print('[결과저장 3]   model.predict:  classification_report  & confusion_matrix')
    (defect_prediction, lacuna_prediction, spoke_prediction, spot_prediction) = model.predict(x_val, batch_size=FLG.BATCH_SIZE)
    prediction = {"defect": defect_prediction, "lacuna": lacuna_prediction, "spoke": spoke_prediction, "spot": spot_prediction}

    f = open(FLG.CONFUSION_MX, 'w')

    for key in classes.keys():
        print('## classification_report')
        classfi_report = classification_report(y_val[key].argmax(axis=1),
                                               prediction[key].argmax(axis=1),
                                               target_names=None) # target_names = [prediction.keys(), 'normal'])

        print(key + ' : ' +classfi_report)
        f.write(key + ' : ' + str(classfi_report))
        print(classes)
        print('## confusion_matrix')
        confu_mx = confusion_matrix(
            y_val[key].argmax(axis=1),
            prediction[key].argmax(axis=1))

        print(confu_mx)
        f.write(str(confu_mx))

        print('## confusion_matrix plot')
        confusion_matrix_saved(confu_mx, classes[key], key)

    f.close()


def show_gragh(hist):
    import matplotlib.pyplot as plt
    import numpy as np
    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "defect_loss", "lacuna_loss", "spoke_output_loss", "spot_output_loss"]
    plt.style.use("ggplot")
    (fig, loss_ax) = plt.subplots(5, 1, figsize=(10, 20))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        print(str(i) + " : " + l)
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        loss_ax[i].set_title(title)
        loss_ax[i].set_xlabel("Epoch #")
        loss_ax[i].set_ylabel("Loss")
        loss_ax[i].plot(np.arange(0, FLG.EPOCHS), hist.history[l], label=l)
        loss_ax[i].plot(np.arange(0, FLG.EPOCHS), hist.history["val_" + l], label="val_" + l)
        loss_ax[i].legend()

    # save the losses figure
    plt.tight_layout()
    plt.savefig(FLG.PLOT_LOSS)
    # plt.show()
    plt.close()

    # create a new figure for the accuracies
    accuracyNames = ["defect_output_acc", "lacuna_output_acc", "spoke_output_acc", "spot_output_acc"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(4, 1, figsize=(10, 16))

    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
        print(str(i) + " : " + l)
        # plot the loss for both the training and validation data
        ax[i].set_title("Accuracy for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Accuracy")
        ax[i].plot(np.arange(0, FLG.EPOCHS), hist.history[l], label=l)
        ax[i].plot(np.arange(0, FLG.EPOCHS), hist.history["val_" + l], label="val_" + l)
        ax[i].legend()

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(FLG.PLOT_ACC)
    # plt.show()
    plt.close()