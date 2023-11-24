import numpy
import numpy as np
import os
import tensorflow as tf
import pathlib
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')

thresh = 0.1
Config_path = "config.txt"
config = open(Config_path).read()
config = config.splitlines()

PATH_TO_MODEL_DIR = config[0]
PATH_TO_LABELS = config[1]
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR
# загрузка модели
detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

def affine_transform(img):
    rows, cols, ch = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    # camera is located not perpendicular to frame, so this code transforms image to somewhat straight
    # this improves detection quality
    pts1 = numpy.float32([[0, rows * 0.05], [cols * 1.05, rows * 0.05], [cols * 0.1, rows * 0.98], [cols * 0.95, rows]])
    pts2 = numpy.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    first_third = int(cols / 3)
    second_third = cols - first_third
    cropped_image1 = dst[0:rows, 0:first_third]
    cropped_image2 = dst[0:rows, first_third:second_third]
    cropped_image3 = dst[0:rows, second_third:cols]
    return cropped_image1, cropped_image2, cropped_image3

def distance(x1, y1, x2, y2):
    return np.sqrt(((x1-x2)**2 + (y1-y2)**2))

def dumb_detection(img):
    image, detections = show_inference(detection_model, img)
    return image

def remove_overlap(output_dict):
    threshold = 0.12
    output_dict1 = output_dict.copy()
    i = 0
    kek = 0
    for scores in output_dict['detection_multiclass_scores']:
        for l, score in enumerate(scores):
            if score < max(scores):
                scores[l] = 0
    for scores in output_dict['raw_detection_scores']:
        for l, score in enumerate(scores):
            if score < max(scores):
                scores[l] = 0
    while i < output_dict['detection_classes'].size-1:
        #print(i)
        #new_scores = np.array()
        #print(output_dict['detection_multiclass_scores'])
        #print(output_dict['detection_multiclass_scores'])
        j = output_dict['detection_classes'].size - 1
        while j > i:
            #print(j)
            #print("i = " + str(i) + " j = " + str(j) + " | " + str(output_dict['detection_classes'].size) + " | " + str(output_dict['raw_detection_boxes'][i]))
            x1, y1, x11, y11 = output_dict['detection_boxes'][i]
            x2, y2, x22, y22 = output_dict['detection_boxes'][j]
            dist1 = distance(x1, y1, x2, y2)
            dist2 = distance(x11, y11, x22, y22)
            if (dist1 + dist2) < threshold:
                #print("dist:")
                #print(dist2 + dist1)
                kek+=1
                if output_dict['detection_scores'][i] < output_dict['detection_scores'][j]:
                    output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], i, 0)
                    output_dict['raw_detection_boxes'] = np.delete(output_dict['raw_detection_boxes'], i, 0)
                    output_dict['raw_detection_scores'] = np.delete(output_dict['raw_detection_scores'], i, 0)
                    output_dict['detection_multiclass_scores'] = np.delete(output_dict['detection_multiclass_scores'], i, 0)
                    output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], i, 0)
                    output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], i, 0)
                    output_dict['detection_anchor_indices'] = np.delete(output_dict['detection_anchor_indices'], i, 0)
                else:
                    output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], j, 0)
                    output_dict['raw_detection_boxes'] = np.delete(output_dict['raw_detection_boxes'], j, 0)
                    output_dict['raw_detection_scores'] = np.delete(output_dict['raw_detection_scores'], j, 0)
                    output_dict['detection_multiclass_scores'] = np.delete(output_dict['detection_multiclass_scores'], j, 0)
                    output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], j, 0)
                    output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], j, 0)
                    output_dict['detection_anchor_indices'] = np.delete(output_dict['detection_anchor_indices'], j, 0)
            j -= 1
        i += 1
    return output_dict


def detect_and_count(img):
    global thresh
    with tf.device('/GPU:0'):
        # model efficientdetd0 can only process images of size 512x512,
        # so I split large input image into 3 parts and process them separately
        # due to specific of task it doesn't create error in detections
        cropped1, cropped2, cropped3 = affine_transform(img)
        image1, detections1 = show_inference(detection_model, cropped1)
        image2, detections2 = show_inference(detection_model, cropped2)
        image3, detections3 = show_inference(detection_model, cropped3)
        dicts = [detections1, detections2, detections3]
        image = cv2.hconcat([image1, image2, image3])
        detection_numbers = dict()
        for detections in dicts:
            for i in range(0, detections['detection_classes'].size):
                if detections['detection_scores'][i] >= thresh:
                    if detection_numbers.get(detections['detection_classes'][i]) is None:
                        detection_numbers.setdefault(detections['detection_classes'][i], 1)
                    else:
                        detection_numbers[detections['detection_classes'][i]] += 1
        return image, detection_numbers

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, frame):
    # take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    image_np=np.compress([True, True, True, False], image_np, axis=2)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    output_dict = remove_overlap(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        max_boxes_to_draw=400,
        min_score_thresh=.2,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=5)

    return (image_np, output_dict)

