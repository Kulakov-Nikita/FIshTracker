import torch


class Sector:
    def __init__(self, center_pos: tuple[int, int], radius: float, image_shape: tuple[int, int]):
        # Создаём координаты для каждого пикселя в тензоре
        x = torch.arange(image_shape[0], dtype=torch.float32)
        y = torch.arange(image_shape[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)

        # Вычисляем расстояние от каждого пикселя до центра окружности
        distance = torch.sqrt((xx - center_pos[1]) ** 2 + (yy - center_pos[0]) ** 2)

        # Создаём маску, где внутри окружности значения 1, а снаружи 0
        self.mask = (distance <= radius).float().to('cuda')

    def intersection(self, image):
        intersection = self.mask * image
        return torch.sum(intersection, dim=(1, 2), keepdim=True)

