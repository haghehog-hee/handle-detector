# импорт необходимых библиотек
import os
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from detection import remove_overlap
import cv2 as cv
# пути к модели и меткам
PATH_TO_MODEL_DIR = "C:\\Tensorflow\\models\\research\\object_detection\\interference_graph\\saved_model"
PATH_TO_LABELS = "C:\\Tensorflow\\Dataset\\label_map2.pbtxt"
IMAGE_SAVE_PATH = "C:\\Tensorflow\\dataset\\autoresult\\"
IMAGE_PATH = "C:\\Tensorflow\\Dataset\\unlabeled\\"
ANNOTATION_SAVE_PATH = "C:\\Tensorflow\\Dataset\\autolabels\\"
canny_img_path = "C:\\Tensorflow\\Dataset\\testimages\\"
canny_ann_path = "C:\\Tensorflow\\Dataset\\testset\\"
canny_img_save = "C:\\Tensorflow\\Dataset\\autoimages\\"
canny_ann_save = "C:\\Tensorflow\\Dataset\\test\\"
flag = True


thresh = 0.3

IMAGE_PATHS = os.listdir(IMAGE_PATH)

# путь к модели
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR
print('Loading model...', end='')
start_time = time.time()

# загрузка модели
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# создание словаря для меток
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# игнорирование предупреждений Matplotlib
warnings.filterwarnings('ignore')


def to_canny(image_path, annotation_path,img_save_path,ann_save_path):
    img_paths = os.listdir(image_path)
    ann_paths = os.listdir(annotation_path)
    for path in img_paths:
        canny_path = img_save_path + "canny" + path
        img = cv.imread(image_path + path)
        if img[0][0][0] == img[0][0][1] and img[0][0][0] == img[0][0][2]:
            alpha = 1
            beta = 1.2
            img = cv.convertScaleAbs(img, alpha, beta)
            img = cv.Canny(img, 10, 20)
        else:
            img = cv.Canny(img, 40, 60)
        cv.imwrite(canny_path, img)
        ann_path = path.replace('.jpg', '.xml')
        canny_ann_path = ann_save_path + "canny" + ann_path
        if ann_path in ann_paths:
            with open(annotation_path + ann_path) as f:
                my_lines = f.readlines()
                my_lines[2] = "\t<filename>"+ path +"</filename>\n"
                my_lines[3] = "\t<filename>"+ canny_path +"</filename>\n"
                file = open(canny_ann_path, "w")
                text = ""
                for line in my_lines:
                    text += line
                file.write(text)
                file.close()

# функция загрузки изображения в numpy массив
def load_image_into_numpy_array(IMAGE_PATH):
    """Загрузка изображения из файла в массив numpy.
    Изображение помещается в массив numpy для передачи в граф tensorflow.
    Обратите внимание, что по соглашению мы помещаем его в массив numpy с формой
    (высота, ширина, каналы), где channels=3 для RGB.
    Args:
        IMAGE_PATH: путь к файлу изображения
    Returns:
        массив numpy uint8, формы (высота, ширина, 3)
    """
    return np.array(Image.open(IMAGE_PATH))
def detection_to_text (detections, filename, image_path, width, height):
    _path = ANNOTATION_SAVE_PATH + filename.replace('.jpg', '.xml')
    file = open(_path, "w")
    _text = '''<annotation>
	<folder>autoannotations</folder>
	<filename>''' + filename + '''</filename>
	<IMAGE_PATH>''' + image_path + '''</IMAGE_PATH>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>'''+str(width)+'''</width>
		<height>'''+str(height)+'''</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>'''
    for i in range(detections['detection_scores'].size):
        box = detections['detection_boxes'][i]
        type = detections['detection_classes'][i]
        text = \
            '''<object> 
                   <name>handle''' + str(type).replace(".0","") + '''</name>
                   <pose>Unspecified</pose> 
                   <truncated>0</truncated> 
                   <difficult>0</difficult> 
                   <bndbox> 
                       <xmin>''' + str(box[1]*width) + '''</xmin> 
                       <ymin>''' + str(box[0]*height) + '''</ymin> 
                       <xmax>''' + str(box[3]*width) + '''</xmax> 
                       <ymax>''' + str(box[2]*height) + '''</ymax> 
                   </bndbox> 
               </object>'''
        _text += text
    _text += '''</annotation>'''
    file.write(_text)
    file.close()
# цикл по всем изображениям

if flag:
    for PATH in IMAGE_PATHS:
        image_path = IMAGE_PATH + PATH
        print('Running inference for {}... '.format(image_path), end='')
        # загрузка изображения
        #image_np = load_image_into_numpy_array(image_path)
        im = Image.open(image_path)
        # if im[0][0][0] == im[0][0][1] and im[0][0][0] == im[0][0][2]:
        #     alpha = 1
        #     beta = 1.2
        #     im2 = cv.convertScaleAbs(im, alpha, beta)
        #     im2 = cv.Canny(im2, 30, 40)
        # else:
        #     im2 = cv.Canny(im, 80, 110)
        # PATH2 = "canny" + PATH
        # image_path2 = IMAGE_PATH + PATH2
        width, height = im.size
        im = im.convert('RGB')
        image_np = np.asarray(im)
        # преобразование в тензор
        input_tensor = tf.convert_to_tensor(image_np)

        # добавление размерности пакета к изображению
        input_tensor = input_tensor[tf.newaxis, ...]

        # выполнение предсказания
        detections = detect_fn(input_tensor)
        # конвертирование тензоров в массив numpy и удаление пакета
        num_detections = int(detections.pop('num_detections'))
        #print(num_detections)

        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections = remove_overlap(detections)
        detections['num_detections'] = num_detections
        #detection_to_text(detections, PATH2, image_path2, width, height)
        detection_to_text(detections, PATH, image_path, width, height)

        # визуализация результатов предсказания на изображении
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=None,
            min_score_thresh=thresh,
            agnostic_mode=False)

        # вывод списка классов, обнаруженных на изображении
        #print(detections['num_detections'])
        d = dict()
        for i in range (0,detections['detection_classes'].size):
            if(detections['detection_scores'][i]>=thresh):
                if(d.get(detections['detection_classes'][i])==None):
                    d.setdefault(detections['detection_classes'][i],1)
                else:
                    d[detections['detection_classes'][i]]+=1
        #print(d)
        # отображение и сохранение изображения с визуализированными результатами
        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.show()

        img = Image.fromarray(image_np_with_detections)
        img.save(IMAGE_SAVE_PATH + "1" + PATH)
        #im2.save(image_path2 + PATH2)
        print('Done')
if not flag:
    to_canny(canny_img_path,canny_ann_path, canny_img_save, canny_ann_save)