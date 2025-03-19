### FishTracker

---

Усовершенствованная верcия [FishSpy](https://github.com/Kulakov-Nikita/FishSpy), адаптированная под требования заказчика (ВолгГМУ).

Является частью программно-аппаратного комплекса, предназанченного для автоматизации проведения экспериментов по системе "Чистое поле".

Установка зависимостей:
```bash
pip install -r requirements.txt
```

Запуск осуществляется командой:
```bash
python main.py
```
Никаких дополнительных настроек проводить не требуется.

Для эксперимента используется стенд (изображение ниже), который подключается к компьютеру, с помощью которого осуществяется запись эксперимента с его последующим анализом в данной программе.

![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/5.png)

Возможности:
* Определяет в какой облати (центральный сектор, транзитный сектор, периферия) находится рыбка и сколько времени она там пробыла. Ниже приведён пример детекции пересечения границ сектора.
  
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/1.png)
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/2.png)
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/3.png)
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/4.png)
  
* Определяет пройденное рыбкой расстояние. Ниже приведён пример опрделения траектории.

  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/6.png)
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/7.png)
  ![](https://github.com/Kulakov-Nikita/FIshTracker/blob/main/Screenshots/8.png)

* Определяет время проведённое в состоянии покоя
