import numpy as np
import jittor.nn as nn
import jittor as jt
from skimage import measure
import numpy

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins=None): # bins is no longer used but kept for compatibility
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0.0
        self.PD = 0.0
        self.target = 0.0

    def update(self, preds, labels):
        score_thresh = 0.5 # Use a fixed threshold
        
        # Convert the whole batch to numpy arrays
        batch_preds_bool = (jt.sigmoid(preds) > score_thresh).data
        batch_labels_int = labels.data.astype('int64')

        # Iterate over each image in the batch
        for i in range(preds.shape[0]):
            predits = np.squeeze(batch_preds_bool[i]).astype('int64')
            labelss = np.squeeze(batch_labels_int[i])

            # Perform 8-connectivity clustering
            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target += len(coord_label)

            # --- Match predicted regions to ground truth regions ---
            
            # Make a copy of predicted regions to safely modify
            remaining_coord_image = list(coord_image)
            num_true_positives = 0

            for true_region in coord_label:
                centroid_label = np.array(true_region.centroid)
                
                # Find the first predicted region within 3 pixels
                match_found = False
                for i, pred_region in enumerate(remaining_coord_image):
                    centroid_image = np.array(pred_region.centroid)
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    
                    if distance < 3:
                        # Match found, consume the predicted region
                        del remaining_coord_image[i]
                        match_found = True
                        break # Move to the next true region
                
                if match_found:
                    num_true_positives += 1

            self.PD += num_true_positives

            # --- Calculate False Alarms ---
            # Any remaining predicted regions are false positives
            false_alarm_pixels = sum(region.area for region in remaining_coord_image)
            self.FA += false_alarm_pixels

    def get(self,img_num):
        Final_FA =  self.FA / ((256 * 256) * img_num) if img_num > 0 else 0.0
        Final_PD =  self.PD / self.target if self.target > 0 else 0.0

        return Final_FA, Final_PD

    def reset(self):
        self.FA  = 0.0
        self.PD  = 0.0
        self.target = 0.0

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (jt.sigmoid(output) > score_thresh).float32()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float32(), axis=1)
    elif len(target.shape) == 4:
        target = target.float32()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float32())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float32())).sum()
    tn = ((1 - predict) * ((predict == target).float32())).sum()
    fn = (((predict != target).float32()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float32(), axis=1)
    elif len(target.shape) == 4:
        target = target.float32()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float32()
    pixel_labeled = (target > 0).float32().sum()
    pixel_correct = (((predict == target).float32())*((target > 0)).float32()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float32()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float32(), axis=1)
    elif len(target.shape) == 4:
        target = target.float32()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float32())

    area_inter, _  = np.histogram(intersection.data, bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.data, bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.data, bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union 