import pyttsx3
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np

def non_max_suppression(obj_class, bboxes, scores, mask, threshold):
    """
    Perform non-maximum suppression to remove redundant overlapping bounding boxes.
    Simply, just keep detection with high confidence.
    Args:
        obj_class (List[str]): List of object class detected.
        bboxes (List[List[int]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
        scores (List[float]): List of confidence scores corresponding to each bounding box.
        threshold (float): Overlap threshold to determine when to suppress a bounding box.
    Returns:
    List[List[int]], List[str], List[float]: List of filtered bbox, object class, and confidence score.
    """
    i = 0
    while i < len(bboxes):
        j = i + 1
        while j < len(bboxes):
            if compute_iou(bboxes[i], bboxes[j]) > threshold:
                if scores[i] >= scores[j]:  # Keep box i if scores are equal
                    bboxes.pop(j)
                    obj_class.pop(j)
                    scores.pop(j)
                    if obj_class[0] != 'hand':
                        mask.pop(j)
                else:
                    bboxes.pop(i)
                    obj_class.pop(i)
                    scores.pop(i)
                    if obj_class[0] != 'hand':
                        mask.pop(i)
                    i -= 1  # Adjust index after removing the current bbox
                    break  # Exit inner loop to re-evaluate i-th bbox
            else:
                j+=1
        i+=1
    return bboxes, obj_class, scores, mask
                
    """
    selected_indices = []
    
    # Sort bounding boxes by confidence scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    while sorted_indices:
        # Select the bounding box with the highest confidence score
        max_index = sorted_indices.pop(0)
        selected_indices.append(max_index)
        
        # Calculate overlap with other bounding boxes
        overlap_indices = []
        for i in sorted_indices:
            if compute_iou(bboxes[max_index], bboxes[i]) > threshold:
                overlap_indices.append(i)
        
        # Remove overlapping bounding boxes from consideration
        sorted_indices = [i for i in sorted_indices if i not in overlap_indices]
    return selected_indices
    """

def compute_iou(bbox1, bbox2):
    """
    Compute the intersection over union (IoU) between two bounding boxes.
    
    Args:
        bbox1 (List[int]): First bounding box represented as [x1, y1, x2, y2].
        bbox2 (List[int]): Second bounding box represented as [x1, y1, x2, y2].
        
    Returns:
        float: Intersection over union (IoU) between the two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    """
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    """
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def eliminate_inside_bbox(obj_class, bboxes, conf, mask): 
    """
    Executed after non_max_supression
    Args:
        bboxes (List[List[int]]): List of bounding box detected
        obj_class (List): List of

    Returns:
        List[List[int]], List: Filtered bounding box, Filtered object class
    """
    i = 0
    while i < len(bboxes):
        j = i + 1
        while j < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]  # bbox1
            a1, b1, a2, b2 = bboxes[j]  # bbox2
            # Check if bbox2 is inside bbox1
            if a1 >= x1 and b1 >= y1 and a2 <= x2 and b2 <= y2:
                bboxes.pop(j)
                conf.pop(j)
                if obj_class[j] != 'hand' and obj_class[j] != 'Text' and obj_class[j] != 'held' and obj_class[j] != 'not_held':
                    mask.pop(j)
                obj_class.pop(j)
            # Check if bbox1 is inside bbox2
            elif x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2:
                bboxes.pop(i)
                conf.pop(i)
                if obj_class[i] != 'hand' and obj_class[i] !='Text' and obj_class[j] != 'held' and obj_class[j] != 'not_held':
                    mask.pop(i)
                obj_class.pop(i)
                i -= 1  # Adjust i since the current bounding box removed
                break  # Exit the inner loop and restart with the new current bbox
            else:
                j += 1
        i += 1
    return bboxes, obj_class, conf, mask

    """
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            x1,y1,x2,y2 = bboxes[i] #bbox1
            a1,b1,a2,b2 = bboxes[j] #bbox2
            #check if bbox2 inside bbox1
            if(a1>x1 and b1>y1 and a2<x2 and b2<y2):
                bboxes.pop(j)
                obj_class.pop(j)
                j-=1 #adjust the inner index loop, since the bboxes was deleted
            #check if bbox1 inside bbox2
            elif(x1>a1 and y1>b1 and x2<a2 and y2<b2):
                bboxes.pop(i)
                obj_class.pop(i)
                i-=1 #adjust the outer index loop, since the bboxes was deleted
                break #restart the outer loop
    return bboxes, obj_class
    """

def draw_confusion_matrix(truth, pred, labels, end_name):
    """
    Procedure to draw confusion matrix and save the plots
    Args:
        truth (List): List of ground truth
        pred (List): List of prediction by model
        labels (List): List of labels
        end_name (float): For filename purposes
    """
    cm = confusion_matrix(truth, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(cmap=plt.cm.Blues)
    plt.figure(figsize=(15, 15))  # Adjust figure size as needed
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())  # Use current axis
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    #metrics
    # Calculate accuracy and precision
    accuracy = accuracy_score(truth, pred)
    report = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    precision_text = "\n".join([f'Class {label}: {report[str(label)]["precision"]:.4f}' for label in labels])
    recall_text = "\n".join([f'Class {label}: {report[str(label)]["recall"]:.4f}' for label in labels])
    f1_text = "\n".join([f'Class {label}: {report[str(label)]["f1-score"]:.4f}' for label in labels])
    # precision_text = report
    # print(report)
    # Add text to plot
    plt.figtext(0.99, 0.5, f'Accuracy: {accuracy:.4f}\nPrecision:\n{precision_text}\nRecall:\n{recall_text}\nF1:\n{f1_text}', 
                horizontalalignment='right', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    #save folder
    save_folder = 'C:\Skripsi\Handheld_Model\eval'  # Replace with your actual folder path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'confusion_matrix_{end_name}.png')
    
    # Save the plot
    plt.savefig(save_path, format='png')
    plt.close()

def calculate_metrics(truth, pred, labels):
    label_type = type(labels[0])
    report = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    accuracy = format(accuracy_score(truth, pred), '.4f')
    # accuracy = format(report['accuracy'], '.4f')
    # precision = [format(report[i]['precision'], '.4f') for i in list(labels.keys())[:-3]]
    # recall = [format(report[i]['recall'], '.4f') for i in list(labels.keys())[:-3]]
    # f1_score = [format(report[i]['r1-score'], '.4f') for i in list(labels.keys())[:-3]]
    labels = [str(i) for i in labels]
    precision = [format(report[i]['precision'], '.4f') for i in labels]
    recall = [format(report[i]['recall'], '.4f') for i in labels]
    f1_score = [format(report[i]['f1-score'], '.4f') for i in labels]
    report = [accuracy, precision, recall, f1_score]
    return report

def draw_plot(metrics, labels):

    #save folder
    save_folder = 'C:\Skripsi\Handheld_Model\eval'  # Replace with your actual folder path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #assigning and converting value
    x = [float(i) for i in metrics['conf_score']]
    precision = [i for i in metrics['precision']]
    recall = [i for i in metrics['recall']]
    f1 = [i for i in metrics['f1-score']]

    # Plotting multiple line graphs (precision)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    for j in range(len(labels)): #
        plt.plot(x, [float(i[j]) for i in precision], marker='o', linestyle='-', label=f'{labels[j]}')

    # Adding labels and title
    plt.xlabel('Conf Threshold')
    plt.ylabel('Precision')
    plt.title('Precision graphs')
    plt.legend()  # Show legend based on labels

    # Save the plot
    save_path = os.path.join(save_folder, f'plot_precision.png')    
    plt.savefig(save_path, format='png')
    plt.close()


    # Plotting multiple line graphs (recall)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    for j in range(len(labels)): #
        plt.plot(x, [float(i[j]) for i in recall], marker='o', linestyle='-', label=f'{labels[j]}')
        
    # Adding labels and title
    plt.xlabel('Conf Threshold')
    plt.ylabel('Recall')
    plt.title('Recall graphs')
    plt.legend()  # Show legend based on labels

    # Save the plot
    save_path = os.path.join(save_folder, f'plot_recall.png')    
    plt.savefig(save_path, format='png')
    plt.close()


    # Plotting multiple line graphs (f1)
    plt.figure(figsize=(10, 10))  # Adjust figure size as needed
    for j in range(len(labels)): #
        plt.plot(x, [float(i[j]) for i in f1], marker='o', linestyle='-', label=f'{labels[j]}')
    
    #annotate and find the biggest f1 each class
    # for i in range(len(x)):
    #     plt.annotate(f'{f1[i]}', (x[i], float(max([j for j in f1][i]))), textcoords="offset points", xytext=(0,10), ha='center')
    text = ""
    for i in range(len(labels)):
        num = max([j[i] for j in f1])
        id = [idx for idx, j in enumerate(f1) if j[i] == num][0]
        text+=f'Class {labels[i]} max f1-score: {num}, at conf score {x[id]}\n'

    plt.figtext(0.99, 0.5, text, 
                horizontalalignment='right', verticalalignment='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Adding labels and title
    plt.xlabel('Conf Threshold')
    plt.ylabel('F1-Score')
    #search for max value
    plt.title('F1-Score graphs')
    plt.legend()  # Show legend based on labels

    # Save the plot
    save_path = os.path.join(save_folder, f'plot_F1-Score.png')    
    plt.savefig(save_path, format='png')
    plt.close()

def el(bboxes):
    i = 0
    while i < len(bboxes):
        j = i + 1
        while j < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]  # bbox1
            a1, b1, a2, b2 = bboxes[j]  # bbox2
            # Check if bbox2 is inside bbox1
            if a1 >= x1 and b1 >= y1 and a2 <= x2 and b2 <= y2:
                bboxes.pop(j)
            # Check if bbox1 is inside bbox2
            elif x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2:
                bboxes.pop(i)
                i -= 1  # Adjust i since the current bounding box removed
                break  # Exit the inner loop and restart with the new current bbox
            else:
                j += 1
        i += 1
    return bboxes