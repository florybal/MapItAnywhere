import sys
import os
import torch

# Ensure repo root is on sys.path so `mapper` package imports work when running
# this script from the scripts/ folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mapper.models import metrics


def make_synthetic(batch=2, classes=3, H_gt=100, W_gt=100, H_pred=256, W_pred=256):
    # Ground truth in NHWC (batch, H, W, C)
    gt = torch.zeros((batch, H_gt, W_gt, classes), dtype=torch.uint8)
    # Set a small square for class 0
    gt[:, 10:30, 10:30, 0] = 1

    # pred: NCHW but larger spatial size
    pred = torch.zeros((batch, classes, H_pred, W_pred), dtype=torch.float32)
    # Put a prediction that overlaps partially
    pred[:, 0, 20:40, 20:40] = 0.8

    # valid_bev: NCHW where channel count doesn't match
    vb = torch.ones((batch, 1, H_pred, W_pred), dtype=torch.uint8)

    # confidence_map: (N, H, W) smaller size
    conf = torch.zeros((batch, H_gt, W_gt), dtype=torch.uint8)

    output = {'output': pred, 'valid_bev': vb}
    data = {'seg_masks': gt, 'confidence_map': conf}
    return output, data


def run_test():
    out, data = make_synthetic()

    iou = metrics.IOU(num_classes=3)
    pa = metrics.PixelAccuracy()
    mAP = metrics.mAP(num_labels=3)

    # Update methods should not raise
    iou.update(out, data)
    pa.update({'output': out['output'], 'valid_bev': out['valid_bev']}, data)
    try:
        mAP.update(out, data)
    except Exception as e:
        print('mAP raised:', e)

    print('IOU states:', iou.tp_observable, iou.fp_observable, iou.fn_observable)
    print('PixelAccuracy:', pa.correct_pixels.item(), pa.total_pixels.item())


if __name__ == '__main__':
    run_test()
