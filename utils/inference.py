from utils.detection_class import detection
from utils.IoU import compute_iou, non_max_suppression, eliminate_inside_bbox, el
import numpy as np
import cv2
from ultralytics import YOLO

def decide(result_obj, result_hand, threshold):
    """
    decide whether the object was held by hand or not by IoU threshold calculation.
    Args: 
        result_obj (class): result for object detection
        result_hand (class): result for hand detection
    
    Returns:
        Boolean, str, List[List[int]]: 0 -> Not held and 1 -> held, name of object, object bbox coordinate
    """
    calculated = {
        'obj_idx':[],
        'iou':[]
    }
    for i in range(len(result_obj.bbox)):
        for j in range(len(result_hand.bbox)):
            iou = compute_iou(result_obj.bbox[i], result_hand.bbox[j])
            if iou > threshold:
                calculated["iou"].append(iou)
                calculated['obj_idx'].append(i)
    # print(result_obj.cls)
    # print(calculated)
    
    #choosing the object with biggest IoU with hand (any hand detected)
    if calculated['iou']!=[]:
        max_iou = max(calculated["iou"])    
        idx = calculated["obj_idx"][calculated["iou"].index(max_iou)]
        return 1, result_obj.cls[idx], result_obj.bbox[idx], result_obj.mask[idx]
        # try:
        #     return 1, result_obj.cls[idx], result_obj.bbox[idx], result_obj.mask[idx]
        # except IndexError:
        #     print(calculated, result_obj, idx)
    else:
        return 0, None, None, None

def detect_file(filename, conf_score, model_obj, model_hand, IoU_threshold):
    """
    Prediction on image, and extract the value for easy manipulation

    Args:
        source (str) = address of filename associated. 
        conf_score (float) = minimum threshold of object confidence. 
        model_obj (class) = model for object
        model_hand (class) = model for hand
        IoU_threshold = for decide rule
    Returns:
        # class, class, class, class = Class of prediction performed inside. (obj, hand, hand, hand)
        # Bool, str, List[float] = boolean of whether held by hand or not, name of object, and bbox of object. 
    """
    obj =  ["backpack", "umbrella", "handbag", "tie", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", 
        "apple", "orange", "broccoli", "carrot", "potted plant", "book", "vase", "scissors", "teddy bear", 
        "hair drier", "toothbrush", "cell phone"]
    #detect object
    obj_result = model_obj.predict(filename, conf=conf_score, verbose=False, iou=0.25) #added iou parameter

    #detect hand
    hand_result = model_hand.predict(filename, conf=0.015, verbose=False, iou=0.55) #added iou parameter

    #extract detection value (bbox, class, conf  score)
    obj_conf, obj_bbox, obj_cls, obj_mask = [],[],[],[]
    for i in obj_result[0]: #results always have len 1, and iteration will go through every detection data.
        #every end will have [0]. So, the data will not append as a list
        if i.names[i.boxes.cls.tolist()[0]] not in obj:
            continue
        else:
            obj_bbox.append(i.boxes.xyxy.tolist()[0])
            obj_cls.append(i.names[i.boxes.cls.tolist()[0]]) #i.boxes.cls only have index key for class name. names are basically dict
            obj_conf.append(i.boxes.conf.tolist()[0])
            obj_mask.append(i.masks.data[0].tolist())
    obj_result = detection(obj_cls, obj_bbox, obj_conf, obj_mask)
    # print(obj_cls, obj_conf, obj_bbox) #debugging
    #extract hand detection value (bbox, class, conf score)
    obj_conf, obj_bbox, obj_cls = [],[],[]
    for i in hand_result[0]:
        if i.names[i.boxes.cls.tolist()[0]] not in 'hand':
            continue
        else:
            obj_bbox.append(i.boxes.xyxy.tolist()[0])
            obj_cls.append(i.names[i.boxes.cls.tolist()[0]]) #i.boxes.cls only have index key for class name. names are basically dict
            obj_conf.append(i.boxes.conf.tolist()[0])
    hand_result = detection(obj_cls, obj_bbox, obj_conf)
    # print(obj_cls, obj_conf, obj_bbox) #debugging

    #post processing
    # NMS -- reinventing issues (soon deleted)
    # obj_result.bbox, obj_result.cls, obj_result.conf, obj_result.mask = non_max_suppression(obj_result.cls, obj_result.bbox, obj_result.conf, obj_result.mask, 0.25)
    # hand_result.bbox, hand_result.cls, hand_result.conf, hand_result.mask = non_max_suppression(hand_result.cls, hand_result.bbox, hand_result.conf, hand_result.mask, 0.55)
    
    #Delete_inside_bbox
    obj_result.bbox, obj_result.cls, obj_result.conf, obj_result.mask = eliminate_inside_bbox(obj_result.cls, obj_result.bbox, obj_result.conf, obj_result.mask)
    hand_result.bbox, hand_result.cls, hand_result.conf, hand_result.mask = eliminate_inside_bbox(hand_result.cls, hand_result.bbox, hand_result.conf, hand_result.mask)

    #debugging
    # print(obj_result.cls, obj_result.bbox, obj_result.conf)
    # print(hand_result.cls, hand_result.bbox, hand_result.conf)

    # print(decide(obj_result, hand_result, IoU_threshold)) #debugging
    return decide(obj_result, hand_result, IoU_threshold)

def mask_img(img, mask):
    """
    Removing background on image

    Args:
    img (array) = array of iamge
    mask (array) = array of mask

    returns:
    array = list of masked (removed background) image
    """
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] != 1: #if not object, make it black
                img[i][j] = [0,0,0]
            else:
                continue
    return img

def detect_text(img, model, conf=0.219):
    """
    Detecting text on img array

    Args:
    img (Array): array of image that want to be detected (read by cv2)
    model (object): YOLO model to detect text (make sure the class name is 'Text' for text, or chage the code itself)

    Returns:
    List, List: list of cropped text image detected on object, list of object bounding box
    """

    #detecting img
    results = model.predict(img, conf=conf, verbose=False, iou=0.6) 

    #get the bbox
    clss, conf, mask, bbox, texts = [],[],[],[],[]
    for i in results:
        if len(i.boxes.cls.tolist())!=0: #ensure there's a detected text'
            if i.names[i.boxes.cls.tolist()[0]] not in 'Text': #make sure the text class is text
                continue
            else:
                bbox.append(i.boxes.xyxy.tolist())
        else:
            continue
    
    
    # text_result = detection(clss, bbox, conf)
    #delete inside bbox
    if len(bbox) > 0: #ensure there's a detection
        # bbox, clss, conf, mask = eliminate_inside_bbox(boxes, bbox, boxes, boxes)
        bbox = el(bbox[0])
    
    #crop the img
    # Ensure the coordinates are within the image bounds
    if len(bbox) > 0:
        for i in bbox:
            x1 = int(max(0, int(i[0])))
            y1 = int(max(0, int(i[1])))
            x2 = int(min(img.shape[1], int(i[2])))
            y2 = int(min(img.shape[0], int(i[3])))
            cropped_image = img[y1:y2, x1:x2]
            texts.append(cropped_image)
    else: #to ensure a same return of value dimension
        bbox.append([])
        texts.append([])
    
    return bbox, texts

def detect_hh(filename, iou=0.25, model="", mode="handheld", conf=0.005):
    if mode=="handheld":
        if model=="":
            model = YOLO('C:\\Skripsi\\Handheld_Model\\models\\yolov9t_hh2.pt')
        result = model.predict(filename, iou=iou, conf=conf, verbose = False)
        obj_conf, obj_bbox, obj_cls = [],[],[]
        for i in result[0]:
            if i.names[i.boxes.cls.tolist()[0]] not in ['held', 'not_held']:
                continue
            else:
                obj_bbox.append(i.boxes.xyxy.tolist()[0])
                obj_cls.append(i.names[i.boxes.cls.tolist()[0]]) #i.boxes.cls only have index key for class name. names are basically dict
                obj_conf.append(i.boxes.conf.tolist()[0])
        if obj_conf != []:
            detection_result = detection(obj_cls, obj_bbox, obj_conf)
            detection_result.bbox, detection_result.cls, detection_result.conf, detection_result.mask = eliminate_inside_bbox(detection_result.cls, detection_result.bbox, detection_result.conf, detection_result.mask )
            max_conf = max(detection_result.conf)
            max_conf_idx = detection_result.conf.index(max_conf)
            if detection_result.cls[max_conf_idx] == 'held':
                return 1, detection_result.bbox[max_conf_idx], max_conf
            else:
                return 0, detection_result.bbox[max_conf_idx], max_conf
        else:
            return 0, "None", "None"
        
    elif mode=="obj":

        obj =  ["backpack", "umbrella", "handbag", "tie", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", 
        "apple", "orange", "broccoli", "carrot", "potted plant", "book", "vase", "scissors", "teddy bear", 
        "hair drier", "toothbrush", "cell phone"]

        if model=="":
            model = YOLO('C:\\Skripsi\\Handheld_Model\\models\\yolov9c-seg.pt')
        obj_result = model.predict(filename, iou=iou, conf=conf, verbose = False) #to reduce hallucination conf is set to 0.15
        
        obj_conf, obj_bbox, obj_cls, obj_mask = [],[],[],[]
        for i in obj_result[0]: #results always have len 1, and iteration will go through every detection data.
            #every end will have [0]. So, the data will not append as a list
            if i.names[i.boxes.cls.tolist()[0]] not in obj:
                continue
            else:
                obj_bbox.append(i.boxes.xyxy.tolist()[0])
                obj_cls.append(i.names[i.boxes.cls.tolist()[0]]) #i.boxes.cls only have index key for class name. names are basically dict
                obj_conf.append(i.boxes.conf.tolist()[0])
                obj_mask.append(i.masks.data[0].tolist())
        if obj_conf != []:
            obj_result = detection(obj_cls, obj_bbox, obj_conf, obj_mask)
            obj_result.bbox, obj_result.cls, obj_result.conf, obj_result.mask = eliminate_inside_bbox(obj_result.cls, obj_result.bbox, obj_result.conf, obj_result.mask)
            max_conf = max(obj_result.conf)
            max_conf_idx = obj_result.conf.index(max_conf)
            return 1, obj_result.cls[max_conf_idx], obj_result.mask[max_conf_idx], max_conf
        else:
            return 0, "None", "None", "None"
    else:
        raise ValueError("Unknown type or mode. handheld or obj only.")