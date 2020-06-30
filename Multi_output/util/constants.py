import tensorflow as tf

# FLG = tf.flags.FLAGS
FLG = tf.compat.v1.flags.FLAGS
#image 기본 정보(CIFAR-10)
tf.compat.v1.flags.DEFINE_integer('WIDTH', 240,
                     'target spatial dimension 너비')
tf.compat.v1.flags.DEFINE_integer('HEIGHT', 240,
                     'target spatial dimension 높이')
tf.compat.v1.flags.DEFINE_integer('DEPTH', 3,
                     'target spatial dimension depth, The CIFAR10 images are RGB.')
dim = 240*240*3
tf.compat.v1.flags.DEFINE_integer('INPUT_DIM', dim,
                     'target spatial dimension 높이, 32*32*3')

# training hyper parameter 기본 정보
# tf.compat.v1.flags.DEFINE_integer('EPOCHS', 300,
#                      '같은 모이고사를 몇번 반복할것인지')
tf.compat.v1.flags.DEFINE_integer('EPOCHS', 100,
                     '같은 모이고사를 몇번 반복할것인지')
tf.compat.v1.flags.DEFINE_integer('BATCH_SIZE', 16,
                     '몇문제 풀어보고 답을 맞출지, 몇개의 샘플로 가중치를 계산하것인지')
tf.compat.v1.flags.DEFINE_integer('PATIENCE', 30,
                     'callback 함수에 있는 EarlyStopping 의 patience, 몇번까지 봐줄것인지')


# todo 프로젝트 이름 변경해야
# Project 기본 정보
tf.compat.v1.flags.DEFINE_string('DATA_MODEL',
                    'iris_pattern_L2_aug_hh',
                    '프로젝트 이름')

tf.compat.v1.flags.DEFINE_string('PROJECT_NAME',
                    FLG.DATA_MODEL + '_' + str(FLG.EPOCHS) + '_' + str(FLG.BATCH_SIZE),
                    '프로젝트 이름 : data + model + epoch + batch size ')

# validation result 저장 경로 '''
tf.compat.v1.flags.DEFINE_string('PLOT',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/plot'+'_'+str(FLG.PROJECT_NAME) + '.png',
                    'plot png 파일 저장할 경로')

tf.compat.v1.flags.DEFINE_string('PLOT_ACC',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/plot_acc'+'_'+str(FLG.PROJECT_NAME) + '.png',
                    'plot png 파일 저장할 경로')

tf.compat.v1.flags.DEFINE_string('PLOT_LOSS',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/plot_loss'+'_'+str(FLG.PROJECT_NAME) + '.png',
                    'plot png 파일 저장할 경로')

tf.compat.v1.flags.DEFINE_string('CONFUSION_MX_PLOT',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/' + str(FLG.PROJECT_NAME),
                    'confusion_matrix 저장 경로')

tf.compat.v1.flags.DEFINE_string('CONFUSION_MX_PLOT_NOM',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/' + str(FLG.PROJECT_NAME) + '_normal',
                    'confusion_matrix 저장 경로')

tf.compat.v1.flags.DEFINE_string('CONFUSION_MX',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/' + str(FLG.PROJECT_NAME) + '.txt',
                    'confusion_matrix 저장 경로')


tf.compat.v1.flags.DEFINE_string('TENSORBOARD',
                    './output/'+FLG.PROJECT_NAME +'/tensorboard',
                    'tensorboard 저장할 경로')

tf.compat.v1.flags.DEFINE_string('CSV_LOGGER',
                    './output/'+FLG.PROJECT_NAME +'/validationReport/'+FLG.PROJECT_NAME +'.csv',
                    'CSV_LOGGER 저장할 경로')

# model 저장 경로
tf.compat.v1.flags.DEFINE_string('H5',
                    './output/'+FLG.PROJECT_NAME +'/modelsaved/h5/',
                    '.h5 형태로 저장할 경로 : for keras')

tf.compat.v1.flags.DEFINE_string('PB',
                    './output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt_pb/',
                    '.ckpt 형태로 저장할 경로 : for tensorflow')

tf.compat.v1.flags.DEFINE_string('CKPT',
                    './output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt/' + FLG.PROJECT_NAME+' weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                    '.ckpt 형태로 저장할 경로 : for tensorflow')













