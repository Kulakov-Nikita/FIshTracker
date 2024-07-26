from video_processing import *


def main():
    print(f"{read_video('input/fenibut_2.mp4', video_duration_sec=300).nbytes / 1024**3:.2f} GiB")


if __name__ == '__main__':
    main()
