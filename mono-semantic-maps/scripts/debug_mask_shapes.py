import torch
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def make_synthetic(batch=2, classes=3, H_gt=100, W_gt=100, H_pred=256, W_pred=256):
    gt = torch.zeros((batch, H_gt, W_gt, classes), dtype=torch.uint8)
    pred = torch.zeros((batch, classes, H_pred, W_pred), dtype=torch.float32)
    vb = torch.ones((batch, 1, H_pred, W_pred), dtype=torch.uint8)
    conf = torch.zeros((batch, H_gt, W_gt), dtype=torch.uint8)
    return gt, pred, vb, conf

def debug():
    gt, pred, vb, conf = make_synthetic()
    print('gt', gt.shape)
    print('pred', pred.shape)
    print('vb', vb.shape)
    print('conf', conf.shape)

    # mimic mAP logic
    gt_nchw = gt.permute(0,3,1,2)
    print('gt_nchw', gt_nchw.shape)

    if pred.ndim == 4 and pred.shape[1] != gt_nchw.shape[1]:
        if pred.shape[-1] == gt_nchw.shape[1]:
            pred2 = pred.permute(0,3,1,2)
        else:
            pred2 = pred
    else:
        pred2 = pred
    print('pred2', pred2.shape)

    # vb handling
    if vb.ndim == 4:
        cand1 = vb.sum(dim=1) > 0
        cand2 = vb.sum(dim=-1) > 0
        print('cand1', cand1.shape, 'cand2', cand2.shape)
        if cand1.shape[1:] == gt_nchw.shape[2:]:
            vb_spatial = cand1
        elif cand2.shape[1:] == gt_nchw.shape[2:]:
            vb_spatial = cand2
        else:
            vb_spatial = cand1
    elif vb.ndim == 3:
        vb_spatial = vb > 0
    else:
        vb_spatial = vb.bool()

    print('vb_spatial before resize', vb_spatial.shape)
    if vb_spatial.shape[1:] != gt_nchw.shape[2:]:
        vb_spatial = torch.nn.functional.interpolate(vb_spatial.unsqueeze(1).float(), size=gt_nchw.shape[2:], mode='nearest').squeeze(1).bool()
    print('vb_spatial after resize', vb_spatial.shape)

    # conf normalization
    c = conf
    if c.ndim == 4 and c.shape[1] == 1:
        c = c.squeeze(1)
    elif c.ndim == 4 and c.shape[-1] == 1:
        c = c.squeeze(-1)
    print('conf after squeeze', c.shape)
    if c.shape[1:] != gt_nchw.shape[2:]:
        c = torch.nn.functional.interpolate(c.unsqueeze(1).float(), size=gt_nchw.shape[2:], mode='nearest').squeeze(1).long()
    print('conf after resize', c.shape)

    observable_mask = torch.logical_and(vb_spatial, c == 0)
    print('observable_mask', observable_mask.shape)

if __name__ == '__main__':
    debug()
