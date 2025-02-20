class detection:
    def __init__(self, cls, bbox, conf, mask=None):
        self.cls = cls
        self.bbox = bbox
        self.conf = conf
        self.mask = mask
    
    def __str__(self):
        for i in range(len(self.cls)):
            print(f"Detection 1 -> {self.cls[i]} with conf score {self.conf[i]}.\nBbox coord: {self.bbox[i]}")

class prediction:
    def __init__(self, pred_list):
        self.status = [i[0] for i in pred_list]
        self.obj_name = [i[1] for i in pred_list]
        self.bbox = [i[2] for i in pred_list]
        self.mask = [i[3] if len(pred_list) == 4 else None for i in pred_list]
        self.lists = pred_list

    def __str__(self):
        print(self.lists)
