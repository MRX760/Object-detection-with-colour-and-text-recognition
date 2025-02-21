from ultralytics import YOLO
import cv2
import numpy as np
from utils.inference import detect_hh
from utils.color_classifier import classify_color, get_rgb, rgb_to_hsv
from utils.OCR import detect_text
import pyttsx3
from Preprocessing_algorithm.retinex.code import retinex
import multiprocessing
from multiprocessing import Manager, Queue
# import timeit
from datetime import datetime
import keyboard
import time

def read_image(img_address):
    ret, frame = cv2.imread(img_address)
    return frame

def detect_handheld(model1, model2, frame_queue, result, ocr_queue, color_queue, lock):
    # result = model.predict(image, iou=0.55, conf=0.5, verbose = False, )
    while True:
        img = frame_queue.get()
        is_hand, hand_bbox, conf = detect_hh(img, model=model1)
        
        if is_hand:
            x1 = int(max(0, hand_bbox[0]))
            y1 = int(max(0, hand_bbox[1]))
            x2 = int(min(img.shape[1], hand_bbox[2]))
            y2 = int(min(img.shape[0], hand_bbox[3]))
            cropped_image = img[y1:y2, x1:x2]
            
            #modify conf score to 0.25 to limit halucination
            is_object, obj_cls, mask, obj_conf = detect_hh(cropped_image, mode='obj', conf = 0.25, model=model2)

            object_dictionary = {
                'backpack': 'ransel',
                'umbrella': 'payung',
                'handbag': 'tas tangan',
                'tie': 'dasie',
                'bottle': 'botol',
                'cup': 'cangkir',
                'fork': 'garpu',
                'knife': 'pisau',
                'spoon': 'sendok',
                'bowl': 'mangkuk',
                'banana': 'pisang',
                'apple': 'apel',
                'orange': 'jeruk',
                'broccoli': 'brokoli',
                'carrot': 'wortel',
                'potted plant': 'tanaman pot',
                'cell phone': 'telepon seluler',
                'book': 'buku',
                'vase': 'vas',
                'scissors': 'gunting',
                'teddy bear': 'beruang teddy',
                'hair drier': 'pengering rambut',
                'toothbrush': 'sikat gigi'
            }
            # print(f"detected object {obj_cls} {obj_conf} ||||| {result['obj_cls']} {result['obj_cls_conf_score']} color stats -> {result['is_processed_by_color']} {result['obj_color']}")
            # print(f"{result['obj_cls']} {result['obj_cls_conf_score']} {result['obj_color']} {result['obj_text']} {result['is_processed_by_ocr']} {result['is_processed_by_color']}")
            if obj_cls != "None": #if any object is detected
                # print('\n#\n')
                with lock: #locking shared var
                    # print('\n!\n')
                    # first condition means if it first execution, second means if we found the better and queue not processed
                    # third condition means it's not first execution and already spoken (processed)
                    if result['obj_cls'] == "" or result['obj_cls_conf_score'] < obj_conf or result['is_spoken'] == True:                     
                        # print('\n@\n')
                        
                        #edit if not same object and not yet processed
                        # if result['obj_cls'] == "" or result['is_spoken'] == True or result['obj_cls'] != object_dictionary[obj_cls] and ocr_queue.full()==True and color_queue.full()==True:
                        if result['obj_cls'] == "" or result['is_spoken'] == True or result['obj_cls'] != obj_cls and ocr_queue.full()==True and color_queue.full()==True:
                            # print('\n$\n')
                            result['obj_cls'] = obj_cls
                            result['obj_cls_conf_score'] = obj_conf
                            result['is_processed_by_color'] = False
                            # result['is_done_processed_by_color'] = False
                            result['is_processed_by_ocr'] = False
                            # result['is_done_processed_by_ocr'] = False
                            result['is_spoken'] = False

                            #empty queue if there is some
                            if ocr_queue.full()==True and color_queue.full()==True:
                                # precaution if lock fails :)
                                try:
                                    ocr_queue.get()
                                    color_queue.get()
                                except:
                                    pass
                            # print('putting color queue')
                            # print(len(cropped_image), len(mask))
                            color_queue.put((cropped_image.copy(), mask.copy()))
                            # print('putting ocr queue')
                            ocr_queue.put(cropped_image.copy())
                            # print('\n$\n')


def classif_color(result, queue):
    while True:
        if result['is_processed_by_color'] == False and queue.empty() == False:
            img, mask = queue.get()
            
            # if img is  None:
            #     break

            mask = np.uint8(mask)
            if len(mask.shape) == 3 and mask.shape[0] >= 1:
                mask = mask[0]  # Assuming mask is in shape (1, height, width), convert to (height, width)
            mask = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]))
            for j in range(len(mask)): #1080
                    for k in range(len(mask[0])): #1920
                        if mask[j][k] != 1:
                            img[j][k] = [0,0,0]
                        else:
                            continue
            # MSRCP
            cropped_img = retinex.retinex_MSRCP(img)

            # get img RGB - input is cropped img with removed background and pre-processed by MSRCP
            first_rgb, second_rgb, third_rgb = get_rgb(cropped_img)
            
            # convert RGB to HSV
            r,g,b = first_rgb
            h,s,v = rgb_to_hsv(r,g,b)
        
            # color classification
            obj_col = classify_color(h,s,v)

            color_dictionary = {
                'white': 'putih',
                'black': 'hitam',
                'red': 'merah',
                'blue': 'biru',
                'yellow': 'kuning',
                'orange': 'oranye',
                'pink': 'merah muda',
                'purple': 'ungu',
                'brown': 'coklat',
                'gray': 'abu-abu',
                'green': 'hijau'
            }
            result['obj_color'] = obj_col
            result['is_processed_by_color'] = True
             
def OCR(result, queue):
    while True:
        if result['is_processed_by_ocr'] == False and queue.empty() == False:
            # print('\n\nOCR begin\n\n')
            img = queue.get()
            if img is None:
                break
            obj_txt = detect_text(img, ['en', 'id'])
            result['obj_text'] = obj_txt
            result['is_processed_by_ocr'] = True

def speak(result):
    # print('initialize tts engine')
    engine = pyttsx3.init()
    # Set the Indonesian voice (Andika)
    # print('initialize tts engine property')

    #andika character voice, caused in Issue where it's not working.
    # engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ID-ID_Andika_11.0')

    # Set properties (optional)
    engine.setProperty('rate', 150)    # Speed of speech (default is 200)
    engine.setProperty('volume', 1.0)  # Volume level (range 0.0 to 1.0)
    # count = 0
    while True:
        # print('checking..')
        if result['is_processed_by_color'] == True and result['is_processed_by_ocr'] == True and result['is_spoken'] == False:
            # print('creating text..')
            # word = result['obj_cls'] + " " + result['obj_color']
            # if result['obj_text'] != "":
            #     word+ " dengan teks " + result['obj_text']
            word = result['obj_color'] + " " + result['obj_cls']
            if result['obj_text'] != "":
                word + " with overlaying text, " + result['obj_text']
            result['is_spoken'] = True
            print(word)
            now = datetime.now()
            print(f"Speak at: {now.strftime("%Y-%m-%d %H:%M:%S")}")  # Example: 2025-02-21 14:35:12
            # engine.say(word)
            # engine.runAndWait()

def read_cam(lock, frame_queue):
    """Continuously reads frames from camera and adds them to queue."""
    cam = None
    for cam_id in [1, 0]:  
        cam = cv2.VideoCapture(cam_id)
        if cam.isOpened():
            break
    if not cam or not cam.isOpened():
        raise ValueError("❌ No camera detected!")

    while True:
        ret, frame = cam.read()
        # print('reading cam')
        if not ret:
            print("❌ Failed to read frame!")
            continue  # Skip this iteration if frame is invalid
        # print('locking..')
        # with lock:
        # print('checking..')
        with lock:
            if frame_queue.full() == True:
                # print('switching frame')
                try:
                    frame_queue.get(timeout=0.5)  # Remove the oldest frame
                except:
                    pass
        frame_queue.put(frame)
        # print('loopin..')
        time.sleep(0.03)  # Prevent CPU overuse (~30 FPS)



if __name__ == "__main__":
    with Manager() as manager:
        lock = multiprocessing.Lock()
        processes = []  
        ocr_queue = manager.Queue(maxsize=1)
        color_classifier_queue = manager.Queue(maxsize=1)
        frame_queue = manager.Queue(maxsize=1)  
        process_interrupt = manager.Value(bool, False)
        
        #variable for controlling flow
        # result = manager.dict({
        #     'obj_cls': manager.Value(str, ""),
        #     'obj_cls_conf_score': manager.Value(float, 0.0),
        #     'obj_color': manager.Value(str, ""),
        #     'obj_text': manager.Value(str, ""),
        #     'is_processed_by_ocr': manager.Value(bool, False),
        #     'is_processed_by_color': manager.Value(bool, False),
        #     'is_done_processed_by_ocr': manager.Value(bool, False),
        #     'is_done_processed_by_color': manager.Value(bool, False),
        #     'is_spoken': manager.Value(bool, False)
        # })
        result = manager.dict({
            'obj_cls': "",
            'obj_cls_conf_score': 0.0,
            'obj_color': "",
            'obj_text': "",
            'is_processed_by_ocr': False,
            'is_processed_by_color': False,
            'is_spoken': False
        })
        
        #yolov9 model 
        try:
            model1 = YOLO("models\\yolov9t_hh.pt")
            model2 = YOLO("models\\yolov9c-seg.pt")
        except:
            try:
                model1 = YOLO("models/yolov9t_hh.pt")
                model2 = YOLO("models/yolov9c-seg.pt")
            except Exception as e:
                raise e

        #child process initialization
        stream_process = multiprocessing.Process(target=read_cam, args=(lock, frame_queue))
        yolov9_process = multiprocessing.Process(target=detect_handheld, args=( model1, model2, frame_queue, result, ocr_queue, color_classifier_queue, lock))
        color_process = multiprocessing.Process(target=classif_color, args=(result, color_classifier_queue))
        ocr_process = multiprocessing.Process(target=OCR, args=(result, ocr_queue))
        # listener_process = multiprocessing.Process(target=key_listener, args=(processes,))
        tts_process = multiprocessing.Process(target=speak, args=(result,))
        
        #starting child process
        now = datetime.now()
        print(f"program begin: {now.strftime("%Y-%m-%d %H:%M:%S")}")  # Example: 2025-02-21 14:35:12
        stream_process.start()
        yolov9_process.start()
        ocr_process.start()
        color_process.start()
        tts_process.start()

        stream_process.join()
        yolov9_process.join()
        ocr_process.join()
        color_process.join()
        tts_process.join()