from video_processing import *
from contours import *
import torch
from matplotlib import pyplot as plt


def main():
    # Загрузка модели на GPU
    model = torch.load("models/torch_eight.pth").to('cuda')

    # Загрузка пакета
    video_np: np.ndarray = read_video(path_to_input_video='input/fenibut_2.mp4', video_duration_sec=3)

    # Отправка пакета на GPU
    video_torch = torch.tensor(video_np, dtype=torch.uint8, device='cuda')
    video_torch = video_torch.to(torch.float32)

    # Обработка пакета
    video_torch = video_torch.permute(0, 3, 1, 2)
    result = model(video_torch)

    # Возврат пакета на CPU
    result = result.to(torch.uint8)
    result = result.cpu()

    # Преобразуем tensor в numpy
    result = result.permute(0, 2, 3, 1)
    result = result.numpy()
    #result = np.stack((result,)*3, axis=3).reshape(result.shape[0], result.shape[1], result.shape[2], 3)

    # Обводим область с рыбкой
    result = draw_contours(background_frames=video_np, mask_frames=result)

    # Запись результата на диск
    write_video(frames=result, path_to_output_video='output/fenibut_2.avi')


if __name__ == '__main__':
    main()
