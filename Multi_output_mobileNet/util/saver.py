import keras
from .constants import *

from tensorflow.python.saved_model import signature_def_utils, signature_constants

class ModelSaver():
    sess = []
    def __init__(self, model):
        self.model = model
        self.sess = tf.compat.v1.keras.backend.get_session()


    def h5saved(self):
        h5_path = FLG.H5 + 'model.h5'
        h5_weights_path = FLG.H5 +'model_weights.h5'
        print(h5_weights_path)
        self.model.save(h5_path)
        self.model.save_weights(h5_weights_path)

        '''
           .h5 형태로 저장 : for keras
          ㄴNAME.h5
        '''


    def ckptsaved(self):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, FLG.PB + FLG.PROJECT_NAME + '.ckpt')
        '''
          .ckpt 형태로 저장 : for tensorflow
          ㄴcheckpoint
          ㄴNAME.ckpt.data-00000-of-00001
          ㄴNAME.ckpt.index
          ㄴNAME.ckpt.meta
        '''


    def pbsaved(self):
        ''' .ckpt -> .pb로 파일 변환하기'''
        builder = tf.saved_model.builder.SavedModelBuilder(FLG.PB + FLG.PROJECT_NAME)  # 체크포인트 파일 불러오기
        builder.add_meta_graph_and_variables(sess=self.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING])
        builder.save()


    def save_tfLite(self):
        ''' 저장한 파일로부터 모델 변환 후 다시 저장 '''
        h5_path = FLG.H5 + FLG.PROJECT_NAME + '.h5'
        tfLite_path= FLG.H5 + FLG.PROJECT_NAME + '.tflite'

        converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
        flat_data = converter.convert()

        with open(tfLite_path, 'wb') as f:
            f.write(flat_data)




