import torch

from intersection import Sector


def remove_outliers(distances: torch.Tensor):
    Q1 = distances.quantile(0.25)
    Q2 = distances.quantile(0.75)
    IQR = Q2 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q2 + 1.5 * IQR

    # Удаляем выбросы
    return distances[torch.logical_and(lower_bound <= distances, distances <= upper_bound)]


class MovementDetector:
    def __init__(self, zone: Sector, image_shape: tuple[int, int], batch_size: int, pixels_per_centimeter: float, frames_num: int):
        self.zone: Sector = zone
        self.batch_size = batch_size
        self.height, self.width = image_shape
        self.pixels_per_centimeter: float = pixels_per_centimeter

        self.distances = torch.zeros(frames_num)
        self.frames_counter: int = 0

        # Создаем координаты для каждой позиции в H и W
        self.y_coords = torch.arange(self.height, dtype=torch.float32, device='cuda:0').view(1, self.height, 1).expand(self.batch_size, self.height, self.width)
        self.x_coords = torch.arange(self.width, dtype=torch.float32, device='cuda:0').view(1, 1, self.width).expand(self.batch_size, self.height, self.width)

    def compute_center_of_mass(self, frames: torch.Tensor) -> torch.Tensor:
        # Оставляем в кадре только исследуемую область
        in_zone_frames = self.zone.intersection(frames)

        # Убираем лишнюю размерность канала (сжимаем с (batch_size, 1, H, W) до (batch_size, H, W))
        in_zone_frames = in_zone_frames.squeeze(1)  # Теперь форма (batch_size, H, W)

        # Вычисляем суммы масок для нормировки (чтобы не было деления на 0)
        frames_sums = in_zone_frames.sum(dim=(1, 2), keepdim=True)  # (batch_size, 1, 1)
        frames_sums[frames_sums == 0] = 1  # Избегаем деления на ноль

        # Вычисляем центры масс по осям Y и X
        y_center = (in_zone_frames * self.y_coords).sum(dim=(1, 2)) / frames_sums.squeeze()  # (batch_size,)
        x_center = (in_zone_frames * self.x_coords).sum(dim=(1, 2)) / frames_sums.squeeze()  # (batch_size,)

        # Объединяем центры масс в тензор (batch_size, 2)
        centers_of_mass = torch.stack([x_center, y_center], dim=1)  # (batch_size, 2)

        return centers_of_mass

    def compute_distances(self, frames) -> None:
        centers_of_mass = self.compute_center_of_mass(frames)
        differences = centers_of_mass[1:] - centers_of_mass[:-1]
        self.distances[self.frames_counter:self.frames_counter+self.batch_size-1] = torch.sqrt(torch.sum(differences ** 2, dim=1))

    def distance(self) -> torch.Tensor:
        return torch.sum(remove_outliers(self.distances)) * self.pixels_per_centimeter

    def time_without_movement(self, threshold: float) -> torch.Tensor:
        return torch.sum(self.distances < threshold)
