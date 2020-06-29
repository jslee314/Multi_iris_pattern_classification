from .constants import *
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import numpy as np

class CustomCallback():
    @staticmethod
    def callback():

        # 1. 텐서보드 콜백함수 정의
        # tensorBoard_hist = TensorBoard(log_dir=FLG.TENSORBOARD,
        #                                                histogram_freq=0,
        #                                                write_graph=True,
        #                                                write_images=True)
        '''
        콘솔에 
        >> cd D:Source/PythonReposit/PyImage/VGG/output/tensorboard        
        >> tensorboard --logdir=.
        명령어를 친다
        '''

        # 2. 조기종료 콜백함수 정의
        early_stopper = EarlyStopping(monitor='val_loss',
                                                       min_delta=0,
                                                       patience=FLG.PATIENCE)


        # 3. 체크포인트 파일 콜백함수 정의
        checkpointer= ModelCheckpoint(filepath=FLG.CKPT,
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True)
        '''
        verbose :  해당함수의 진행사항의 출력여부
        save_best_only : 모델의 정확도가 최고값을 갱신했을 때만 저장
        에포크 후에 모델을 저장하십시오. 
        '''

        # 4. 학습속도 조절
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=10,
                                       min_lr=0.5e-6)
        ''' 
        측정 항목이 개선되지 않을 때 학습 속도를 줄입니다. 
        모델은 학습이 정체되면 학습 속도를 2 ~ 10 배 정도 줄이는 것이 좋습니다. 
        이 콜백은 수량을 모니터링하고 '인내'수의 에포크가 개선되지 않으면 학습 속도가 감소합니다.
        '''

        # 5. 에포크 결과를 CSV 파일로 저장
        # csv_logger = CSVLogger(FLG.CSV_LOGGER)
        ''' 
        에포크 결과를 CSV 파일로 스트리밍하는 콜백입니다. 
        np.ndarray와 같은 1D iterables를 포함하여 문자열로 표현할 수있는 모든 값을 지원합니다.
        '''
        list_callbacks = [early_stopper, checkpointer,  lr_reducer]

        # list_callbacks = [tensorBoard_hist, early_stopper, checkpointer, lr_reducer, csv_logger]
        return list_callbacks