import torch
from matplotlib import pyplot as plt


class IntersectionDetector:
    def __init__(self, center_pos: tuple[int, int], center_sector_radius: float, middle_sector_radius: float,
                 outer_sector_radius: float, periphery_radius: float, image_shape: tuple[int, int],
                 threshold: float) -> None:
        self.center_sector: Sector = Sector(center_pos, center_sector_radius, image_shape)
        self.middle_sector: Sector = Sector(center_pos, middle_sector_radius, image_shape)
        self.outer_sector: Sector = Sector(center_pos, outer_sector_radius, image_shape)
        self.periphery: Sector = Sector(center_pos, periphery_radius, image_shape)

        self.center_sector_frame_num: int = 0
        self.middle_sector_frame_num: int = 0
        self.outer_sector_frame_num: int = 0
        self.periphery_sector_frame_num: int = 0

        self.threshold: float = threshold

    def intersection(self, frames) -> None:
        in_center_fish_area = self.center_sector.intersection_area(frames)
        in_middle_fish_area = self.middle_sector.intersection_area(frames)
        in_outer_fish_area = self.outer_sector.intersection_area(frames)
        on_periphery_fish_area = self.periphery.intersection_area(frames)

        self.periphery_sector_frame_num += torch.sum((on_periphery_fish_area - in_outer_fish_area > self.threshold))
        self.outer_sector_frame_num += torch.sum((in_outer_fish_area - in_middle_fish_area > self.threshold))
        self.middle_sector_frame_num += torch.sum((in_middle_fish_area - in_center_fish_area > self.threshold))
        self.center_sector_frame_num += torch.sum(in_center_fish_area > self.threshold)

    def draw_plot(self, frames) -> None:
        intersection = self.middle_sector.intersection(frames) - self.center_sector.intersection(frames)
        intersection = intersection.cpu().detach().numpy()
        plt.plot(range(len(intersection)), intersection)
        plt.show()


class Sector:
    def __init__(self, center_pos: tuple[int, int], radius: float, image_shape: tuple[int, int]) -> None:
        # Создаём координаты для каждого пикселя в тензоре
        x = torch.arange(image_shape[0], dtype=torch.float32)
        y = torch.arange(image_shape[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)

        # Вычисляем расстояние от каждого пикселя до центра окружности
        distance = torch.sqrt((xx - center_pos[1]) ** 2 + (yy - center_pos[0]) ** 2)

        # Создаём маску, где внутри окружности значения 1, а снаружи 0
        self.mask = (distance <= radius).float().to('cuda')

    def intersection(self, frames):
        return self.mask * frames

    def intersection_area(self, frames):
        intersection = self.intersection(frames)
        return torch.sum(intersection, dim=(2, 3), keepdim=True).reshape(len(frames))

