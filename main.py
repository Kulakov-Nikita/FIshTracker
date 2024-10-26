from video_processing import *
from contours import *
import torch
from intersection import *
from movement import MovementDetector
from matplotlib import pyplot as plt


def print_info(intersection_detector, movement_detector, result):
    print("Пакет обработан")

    intersection_detector.intersection(result)

    print(f"Время в центральном секторе: {intersection_detector.center_sector_frame_num / 30 :.1f}s")
    print(f"Время в среднем секторе: {intersection_detector.middle_sector_frame_num / 30 :.1f}s")
    print(f"Время во внешнем секторе секторе: {intersection_detector.outer_sector_frame_num / 30 :.1f}s")
    print(f"Время на перферии: {intersection_detector.periphery_sector_frame_num / 30 :.1f}s")

    movement_detector.compute_distances(result)

    print(f"Длина пройденная рыбкой: {movement_detector.distance():.1f}cm")
    print(f"Время без движения: {movement_detector.time_without_movement(2) / 30:.1f}s")


def print_info_end(intersection_detector, movement_detector):
    print(f"Время в центральном секторе: {intersection_detector.center_sector_frame_num / 30 :.1f}s")
    print(f"Время в среднем секторе: {intersection_detector.middle_sector_frame_num / 30 :.1f}s")
    print(f"Время во внешнем секторе секторе: {intersection_detector.outer_sector_frame_num / 30 :.1f}s")
    print(f"Время на перферии: {intersection_detector.periphery_sector_frame_num / 30 :.1f}s")

    print(f"Длина пройденная рыбкой: {movement_detector.distance():.1f}cm")
    print(f"Время без движения: {movement_detector.time_without_movement(2) / 30:.1f}s")

def main():
    # Настройки
    sector_center = (610, 400)
    center_sector_radius = 45
    middle_sector_radius = 125
    outer_sector_radius = 225
    periphery_radius = 280
    image_shape = (720, 1280)
    batch_size = 90
    pixels_per_centimeter = 1

    # Загрузка модели на GPU
    model = torch.load("models/torch_eight.pth").to('cuda')
    model.eval()

    intersection_detector = IntersectionDetector(sector_center, center_sector_radius, middle_sector_radius,
                                                 outer_sector_radius, periphery_radius, image_shape,
                                                 threshold=0)

    movement_detector = MovementDetector(zone=intersection_detector.periphery, image_shape=image_shape,
                                         batch_size=batch_size, pixels_per_centimeter=pixels_per_centimeter,
                                         frames_num=batch_size)

    video_reader = VideoReader(path_to_input_video='input/fenibut_2.mp4', image_shape=image_shape, batch_size=batch_size)

    counter = 0

    for batch in video_reader.read_batch():
        batch_torch = torch.tensor(batch, device="cuda:0").to(torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            result = model(batch_torch)

        intersection_detector.intersection(result)
        movement_detector.compute_distances(result)

        del result
        del batch_torch
        torch.cuda.empty_cache()
        print(counter)
        counter += 1

    print_info_end(intersection_detector, movement_detector)

    # # Загрузка пакета
    # video_np: np.ndarray = read_video(path_to_input_video='input/fenibut_2.mp4', video_duration_sec=9)
    #
    # # Отправка пакета на GPU
    # video_torch = torch.tensor(video_np[90:180], dtype=torch.uint8, device='cuda')
    # video_torch = video_torch.to(torch.float32)
    #
    # # Обработка пакета
    # video_torch = video_torch.permute(0, 3, 1, 2)
    #
    # with torch.no_grad():
    #     result = model(video_torch)
    #
    # #print_info(intersection_detector, movement_detector, result)
    # print('done')
    # del video_torch
    # del result
    # torch.cuda.empty_cache()
    # input()

    # video_torch = torch.tensor(video_np[180:270], dtype=torch.uint8, device='cuda')
    # video_torch = video_torch.to(torch.float32)




if __name__ == '__main__':
    main()
