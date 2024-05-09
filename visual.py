import matplotlib.pyplot as plt
import numpy as np
import os
import torch
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

# # sample image 16, query image floating
def adjust_image(image):
    # 이미지 차원을 확인하고 적절히 조정
    if image.dim() == 3:
        return image.cpu().numpy()  # 이미지가 이미 적절한 차원이면 변환만 수행
    elif image.dim() == 4:
        # 이미지 배치인 경우 첫 번째 이미지만 사용
        return image[0].cpu().numpy()
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

def visualize_predictions(sample_images, sample_labels, query_image, query_label, y_preds, batch_index, save_dir):
    fig, axs = plt.subplots(4, 5, figsize=(25, 20))  # 4x5 그리드로 변경, 충분한 크기 확보

    query_image_np = adjust_image(query_image)
    axs[0, 0].imshow(query_image_np.transpose(1, 2, 0))  # 왼쪽 상단에 쿼리 이미지 배치
    axs[0, 0].set_title(f'Query Image - Label: {query_label}', fontsize=10)
    axs[0, 0].axis('off')

    for i in range(1, 4):
        axs[i, 0].axis('off')

    image_index = 0
    for i in range(4):
        for j in range(1, 5):
            if image_index < len(sample_images):
                sample_image_batch = sample_images[image_index]
                # 여기서 라벨을 숫자로 변환
                label_value = sample_labels[image_index].item()  # 수정된 부분
                y_pred = y_preds[image_index]
                sample_image_np = adjust_image(sample_image_batch)
                axs[i, j].imshow(sample_image_np.transpose(1, 2, 0))
                result = 'Match' if y_pred > 0.5 else 'Mismatch'
                axs[i, j].set_title(f'Label: {label_value}\n{result} (Score: {y_pred:.2f})', fontsize=10)
                axs[i, j].axis('off')
                image_index += 1


    save_path = os.path.join(save_dir, 'predictions')
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'batch_{batch_index}_comparison.jpg')
    plt.savefig(file_path)
    plt.close(fig)

    print(f"Saved prediction comparison for batch {batch_index} at {file_path}")
