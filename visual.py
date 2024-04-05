import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_prediction(x1, x2, y_pred, index, save_dir):
    """
    Visualizes the prediction for a pair of images and saves it in the specified directory.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
    ax[0].set_title('Anchor Image')
    ax[0].axis('off')

    ax[1].imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
    # y_pred 값에 따라 'Positive' 또는 'Negative'로 표시
    image_title = 'Positive Image' if y_pred == 0 else 'Negative Image'
    ax[1].set_title(image_title)
    ax[1].axis('off')

    # 디렉토리 생성 (이미 있으면 무시)
    save_path = os.path.join(save_dir, 'prediction')  # 디렉토리 경로 구성
    os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성

    plt.suptitle(f'Prediction: {"Same" if y_pred == 0 else "Different"}')

    # 저장할 파일 경로 구성
    file_path = os.path.join(save_path, f'predic_{index}.jpg')
    plt.savefig(file_path)  # 지정된 경로에 이미지 저장
    plt.close(fig)  # 리소스 해제
