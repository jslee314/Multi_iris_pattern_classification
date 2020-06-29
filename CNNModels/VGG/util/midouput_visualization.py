''' 5.4 컨브넷 학습 시각화

 --- 대표적인 세가지 기법 ---

1. 컨프넷 중간층의 "출력"(중간층에 있는 활성화)을 시각화하기
        연속된 컨브넷 층이 '입력을 어떻게 변형시키는' 이해하고
        '개별적인 컨브넷 필터의 의미'를 파악하는데 도움이 됩니다.

2. 컨브넷 "필터"를 시각화하기
        컨브넷의 '필터가 찾으려는 시각적인 패턴과 개념'이 무엇인지 상세하게 이해하는데 도움이됩니다.

3. "클래스 활성화"에 대한 히트맵(heatmap)을 이미지에 시각화하기
        이미지의 어느 부분이 주어진 클래스에 속하는데 기여했는지를 이해하고
        이미지에서 객체 위치를 추정(localization)하는데 도움이 된다.
'''

# todo (5. 4. 1) 중간층의 활성화 시각화

''' 1.  컨프넷 중간층의 "출력"(중간층에 있는 활성화)을 시각화하기  - 
        중간층의 활성화 시각화는 어떤 입력이 주어졌을 때 네트워크에 있는 
        연속된 컨브넷 층이 '입력을 어떻게 변형시키는' 이해하고
        '개별적인 컨브넷 필터의 의미'를 파악하는데 도움이 됩니다.
'''

from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from keras import models

# 이전에 만든 모델 로드
model = load_model('D:/Source/PythonReposit/VGG/output/Cifar10_VGG16v1_300_20/modelsaved/h5/Cifar10_VGG16v1_300_20.h5')  # ./Cifar10_VGG16v1_300_20/modelsaved/h5/Cifar10_VGG16v1_300_20.h5'
model.summary()

# [5-25]  개별 이미지 전처리하기
img = image.load_img('D:/Data/cifar10/test/0/aeroplane_s_000022.png', target_size=(32, 32))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)     # 이미지를 4D 텐서로 변경  img_tensor (1, 32, 32, 3)
img_tensor /= 255.                                                  # 모델이 훈련될 때 입력에 적용한 전처리방식을 동일하게 사용
print(img_tensor.shape)

# [5-26] 테스트 사진 출력하기
plt.imshow(img_tensor[0])
#plt.show()

# [5-27] 입력텐서와 출력 텐서의 리스트로 모델 인스턴스 만들기
# 확인하고 싶은 특성 맵을 추출하기위해 이미지 배치를 입력으로 받아 모든 합성곱과 풀링층의 활성화를 출력하는 케라스 모델을 만들자
layer_outputs = [layer.output for layer in model.layers[:8]]        # 상위 8개 층을 추출
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # 층의 활성화마다 하나씩 8개의 넘파이 배열로 이루어진 리스트를 반환

# [5-28] 예측 모드로 모델 실행하기
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape) # (1, 32, 32, 3)

# [5-29] 1번째, 2번째, 3번째 채널
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis' )
plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis' )
plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis' )
''' 3개의 각 채널은 다른 피처를 감지하도록 인코딩 되어있다.'''
# plt.show()

# [5-31] 중간층의 모든 활성화에 있는 채널 시각화하기
''' 층의 이름을 그래프의 제목으로 함 '''
layer_names = []
for layer in model.layers[:5]:
    layer_names.append(layer.name)

images_per_row = 16

''' 특성맵을 그림 8개 층에 대하여 '''
for layer_name, layer_activation in zip(layer_names, activations):
    ''' 특성 맵에 있는 특성의 수'''
    n_features = layer_activation.shape[-1]

    ''' 특성맵의 크기는 (1, size.size.n_features)이다. '''
    size = layer_activation.shape[1]

    ''' 활성화 채널을 위한 그리드 크기를 구함 '''
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    ''' 각 활성화를 하나의 큰 그리드에 채움 '''
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            ''' 그래프로 나타내기 좋게 특성을 처리:
               임의의 실수인 층의 출력을 픽셀로 표현이 가능한 0-255 사이의 정수로 바꿈,
               먼저 평균을 배고 표준편차로 나누어 표준 점수(standard score)로 바꾼다. 
               그 다음 표준점수 2.0 이내의 값(약 95%포함됨) 들의 0-255 사이에 놓이도록 증폭시킨 후 클리핑함  
            '''
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image *= 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            ''' 그리드 출력 '''
            display_grid[col * size: (col + 1) * size,
                              row * size: (row+1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                               scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

plt.show()


'''
첫번째 층은 여러 종류의 엣지 감지기를 모아놓은것같다. 
이 단계의 활성화에는 초기 사진에 있는 거의 모든 정보가 유지된다.

상위층으로 갈 수로 활성화는 점점 더 추상적으로 되고 시각적으로 이해하기 어려워진다.
고양이 귀와 고양이 눈 처럼 고수준의 개념을 인코딩하기 시작한다
상휘층의 표현은 이미지의 시각적 콘텐츠에 관한 정보가 점점 줄어들고 
이미지의 클래스에 관한 정보가 점점 증가한다.

비어있는 활성화가 층이 깊어짐에 따라 늘어난다. 
첫번째 층에서는 모든 필터가 입력이미지에 활성화되었지만 
층을 올라가면서 활성화되지 않는 필터들이 생긴다. 
필터에 인코딩된 패턴이 입력이미지에 나타나지 않앗다는 것을 의미한다.
'''


''' 
클래스  활성화 시각화

'''











