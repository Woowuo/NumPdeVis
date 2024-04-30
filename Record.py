import cv2
import os

# 图片所在文件夹路径

image_folder = "./"
# 输出视频的路径
video_path = './static/output_video.mp4'

image_folder = "/Users/24k2/Desktop/study/intro/2402/软件项目管理/NumPdeVis/NumPdeVis"
# 输出视频的路径
video_path = 'output_video.mp4'

# 视频的帧率
fps = 24

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# 按文件名排序
print(images)
images.sort(key=lambda x: int(x.split('.')[0]))
print(images)

# 确保至少有一张图片
if images:
    # 读取第一张图片以确定分辨率
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    frame_size = (width, height)

    # 创建视频编写器对象
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 改用H.264编码
    video = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()

    # 删除图片
    for image in images:
        img_path = os.path.join(image_folder, image)
        os.remove(img_path)

    print("视频生成完毕，所有图片已删除。")
else:
    print("没有找到图片")
