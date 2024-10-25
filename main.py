from video_processing import *
from contours import *
import torch
from intersection import *
from movement import MovementDetector
from matplotlib import pyplot as plt


def main():
    # Настройки
    sector_center = (610, 400)
    center_sector_radius = 45
    middle_sector_radius = 125
    outer_sector_radius = 225
    periphery_radius = 280
    batch_size = 90
    pixels_per_centimeter = 1

    # Загрузка модели на GPU
    model = torch.load("models/torch_eight.pth").to('cuda')

    # Загрузка пакета
    video_np: np.ndarray = read_video(path_to_input_video='input/fenibut_2.mp4', video_duration_sec=6)

    image_shape = video_np.shape[1:-1]

    intersection_detector = IntersectionDetector(sector_center, center_sector_radius, middle_sector_radius,
                                                 outer_sector_radius, periphery_radius, image_shape,
                                                 threshold=0)

    movement_detector = MovementDetector(zone=intersection_detector.periphery, image_shape=image_shape,
                                         batch_size=batch_size, pixels_per_centimeter=pixels_per_centimeter,
                                         frames_num=batch_size)

    # Отправка пакета на GPU
    video_torch = torch.tensor(video_np[90:], dtype=torch.uint8, device='cuda')
    video_torch = video_torch.to(torch.float32)

    # Обработка пакета
    video_torch = video_torch.permute(0, 3, 1, 2)
    result = model(video_torch)

    print("Пакет обработан")

    intersection_detector.intersection(result)

    print(f"Время в центральном секторе: {intersection_detector.center_sector_frame_num / 30 :.1f}s")
    print(f"Время в среднем секторе: {intersection_detector.middle_sector_frame_num / 30 :.1f}s")
    print(f"Время во внешнем секторе секторе: {intersection_detector.outer_sector_frame_num / 30 :.1f}s")
    print(f"Время на перферии: {intersection_detector.periphery_sector_frame_num / 30 :.1f}s")

    movement_detector.compute_distances(result)

    print(f"Длина пройденная рыбкой: {movement_detector.distance():.1f}cm")
    print(f"Время без движения: {movement_detector.time_without_movement(2) / 30:.1f}s")

    #result = intersection_detector.outer_sector.mask + result

    # Возврат пакета на CPU
    result = result.to(torch.uint8)
    result = result.cpu()

    # Преобразуем tensor в numpy
    result = result.permute(0, 2, 3, 1)
    result = result.numpy()
    # result = np.stack((result,)*3, axis=3).reshape(result.shape[0], result.shape[1], result.shape[2], 3)

    # Обводим область с рыбкой
    result = draw_contours(background_frames=video_np[60:], mask_frames=result)

    # Запись результата на диск
    write_video(frames=result, path_to_output_video='output/fenibut_2.avi')


if __name__ == '__main__':
    main()
