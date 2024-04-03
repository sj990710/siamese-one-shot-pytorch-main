import matplotlib
import os

matplotlib.use('Agg')  # Set the matplotlib backend to 'Agg' for non-GUI environments
import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(x1, x2, y_pred, index, save_dir):
    """
    Visualizes the prediction for a pair of images and saves it as a PNG file.

    Args:
    - x1: Tensor, the first image in the pair.
    - x2: Tensor, the second image in the pair.
    - y_pred: int, the predicted label for the pair (0 for 'same', 1 for 'different').
    - index: int, the index of the current prediction in the sequence of predictions.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
    ax[0].set_title('Image 1')
    ax[0].axis('off')

    ax[1].imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
    ax[1].set_title('Image 2')
    ax[1].axis('off')

    # 디렉토리가 없으면 생성, 이미 있으면 무시
    save_path = os.path.join(save_dir, 'predict_images')  # 디렉토리 경로 구성
    os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성

    plt.suptitle(f'Prediction: {"Same" if y_pred == 0 else "Different"}')
    plt.savefig(os.path.join(save_path, f'prediction_{index}.png'))  # 이미지 저장
    plt.close(fig)  # 리소스 해제
