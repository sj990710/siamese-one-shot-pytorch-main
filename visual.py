import matplotlib.pyplot as plt
import numpy as np
import os

# def visualize_prediction(x1, x2, y_pred, anchor_label, index, save_dir):
#     """
#     Visualizes the prediction for a pair of images and saves it in the specified directory.
#     Depending on the prediction result, images are saved in 'same' or 'different' folders.
#     The anchor image is also labeled with its corresponding label.
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
#     ax[0].set_title(f'Anchor Image\nLabel: {anchor_label}')
#     ax[0].axis('off')
#
#     ax[1].imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
#     # y_pred 값에 따라 'Positive' 또는 'Negative'로 표시
#     image_title = 'Positive Image' if y_pred == 0 else 'Negative Image'
#     ax[1].set_title(image_title)
#     ax[1].axis('off')
#
#     # 'same' 또는 'different' 폴더 경로 설정
#     result_folder = 'same' if y_pred == 0 else 'different'
#     save_path = os.path.join(save_dir, 'prediction', result_folder)
#     # 디렉토리 생성 (이미 있으면 무시)
#     os.makedirs(save_path, exist_ok=True)
#
#     plt.suptitle(f'Prediction: {"Same" if y_pred == 0 else "Different"}')
#
#     # 저장할 파일 경로 구성
#     file_path = os.path.join(save_path, f'predic_{index}.jpg')
#     plt.savefig(file_path)  # 지정된 경로에 이미지 저장
#     plt.close(fig)  # 리소스 해제

# def visualize_prediction(x1, x2, y_pred, similarity_label, anchor_label, x2_label, index, save_dir):
#     """
#     Visualizes the prediction for a pair of images and saves it in the specified directory.
#     Depending on the prediction result, images are saved in 'same' or 'different' folders.
#     Both the anchor and the positive/negative images are labeled with their corresponding labels.
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
#     ax[0].set_title(f'Anchor Image\nLabel: {anchor_label}')
#     ax[0].axis('off')
#
#     ax[1].imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
#     # y_pred 값에 따라 'Positive' 또는 'Negative'로 표시
#     image_title = 'Positive Image' if y_pred == 0 else 'Negative Image'
#     # x2 (Positive/Negative) 이미지의 레이블을 제목에 추가
#     ax[1].set_title(f'{image_title}\nLabel: {x2_label}')
#     ax[1].axis('off')
#
#     # 'same' 또는 'different' 폴더 경로 설정
#     result_folder = 'same' if y_pred == 0 else 'different'
#     save_path = os.path.join(save_dir, 'prediction', result_folder)
#     # 디렉토리 생성 (이미 있으면 무시)
#     os.makedirs(save_path, exist_ok=True)
#
#     plt.suptitle(f'Prediction: {"Same" if y_pred == 0 else "Different"}')
#
#     # 저장할 파일 경로 구성
#     file_path = os.path.join(save_path, f'predic_{index}.jpg')
#     plt.savefig(file_path)  # 지정된 경로에 이미지 저장
#     plt.close(fig)  # 리소스 해제

def visualize_prediction(x1, x2, y_pred, similarity_label, anchor_label, x2_label, index, save_dir):
    """
    Visualizes the prediction for a pair of images and saves it in the specified directory.
    Depending on the prediction result, images are saved in 'same' or 'different' folders.
    Both the anchor and the positive/negative images are labeled with their corresponding labels.
    Additionally, similarity label is added to the image title, and y_pred value is displayed between the images.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # axs 배열 확장

    # Anchor 이미지 및 라벨
    axs[0].imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
    axs[0].set_title(f'Anchor Image\nLabel: {anchor_label}')
    axs[0].axis('off')

    # y_pred 값을 중앙에 표시
    axs[1].text(0.5, 0.5, f'Prediction: {"Same" if y_pred == 0 else "Different"}\nSimilarity Score: {similarity_label}',
                verticalalignment='center', horizontalalignment='center', fontsize=12, color='red')
    axs[1].axis('off')

    # x2 이미지 및 라벨
    axs[2].imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
    image_title = 'Positive Image' if y_pred == 0 else 'Negative Image'
    axs[2].set_title(f'{image_title}\nLabel: {x2_label}')
    axs[2].axis('off')

    # 결과 폴더 설정
    result_folder = 'same' if y_pred == 0 else 'different'
    save_path = os.path.join(save_dir, 'prediction', result_folder)
    os.makedirs(save_path, exist_ok=True)

    # 저장 경로 및 파일 이름 설정
    file_path = os.path.join(save_path, f'predic_{index}.jpg')
    plt.savefig(file_path)
    plt.close(fig)
