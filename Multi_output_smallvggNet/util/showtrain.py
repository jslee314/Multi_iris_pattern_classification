from .constants import *
import itertools
import numpy as np
import matplotlib.pyplot as plt


def hist_saved(hist, output_list):
    lossNames = output_list

    plt.style.use("ggplot")
    fig, loss_ax = plt.subplots(4, 1, figsize=(5, 20))
    for (i, l) in enumerate(lossNames):
        acc_ax = loss_ax[i].twinx()

        loss_ax[i].plot(hist.history[l+'_loss'], 'y', label=l+'_loss')
        loss_ax[i].plot(hist.history['val_'+l+'_loss'], 'r', label='val_'+l+'_loss')

        acc_ax.plot(hist.history[l+'_accuracy'], 'b', label=l+'_accuracy')
        acc_ax.plot(hist.history['val_'+l+'_accuracy'], 'g', label='val_'+l+'_accuracy')

        loss_ax[i].set_xlabel('epoch')
        loss_ax[i].set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')
        loss_ax[i].legend(loc='upper left')
        acc_ax.legend(loc='lower left')

    # plt.show()
    plt.savefig(FLG.PLOT)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def confusion_matrix_saved(confu_mx, classes, key):
    # Compute confusion matrix : confu_mx
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confu_mx, classes=classes,
                          title='Confusion matrix, without normalization')
    plt.savefig(FLG.CONFUSION_MX_PLOT +key+  '.png')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confu_mx, classes=classes, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(FLG.CONFUSION_MX_PLOT_NOM+key + '.png')
    # plt.show()
