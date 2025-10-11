import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification


class PixelAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct_pixels", default=torch.tensor(
            0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, pred, data):
        # Normalize GT to NCHW
        gt = data["seg_masks"]
        # If GT is NHWC (last dim == channel count), convert to NCHW
        if gt.ndim == 4 and gt.shape[-1] == pred['output'].shape[1]:
            gt = gt.permute(0, 3, 1, 2)

        pred_out = pred['output']
        # Ensure pred_out is NCHW
        if pred_out.ndim == 4 and pred_out.shape[1] != gt.shape[1]:
            # If pred_out is NHWC (last dim == channel count), convert to NCHW
            if pred_out.shape[-1] == gt.shape[1]:
                pred_out = pred_out.permute(0, 3, 1, 2)

        # Resize pred to GT size if needed
        if pred_out.shape[2:] != gt.shape[2:]:
            pred_out = F.interpolate(pred_out, size=gt.shape[2:], mode='bilinear', align_corners=False)

        output_mask = pred_out > 0.5
        gt_mask = gt

        # Compare only the semantic channels (exclude visibility/extra channel if present)
        if gt_mask.shape[1] > output_mask.shape[1]:
            gt_mask = gt_mask[:, :output_mask.shape[1]]

        self.correct_pixels += ((output_mask == gt_mask).sum())
        self.total_pixels += gt_mask.numel()

    def compute(self):
        return self.correct_pixels / self.total_pixels


class IOU(torchmetrics.Metric):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("tp_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("tp_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, output, data):

        gt = data["seg_masks"]
        pred = output['output']
        # Normalize GT to NCHW (batch, classes, H, W)
        # If GT is NHWC (last dim == channel count), convert to NCHW
        if gt.ndim == 4 and gt.shape[-1] == pred.shape[1]:
            gt = gt.permute(0, 3, 1, 2)

        # Ensure pred is float and has shape NCHW
        if pred.ndim == 4 and pred.shape[1] != gt.shape[1]:
            # If pred was NHWC, convert
            if pred.shape[-1] == gt.shape[1]:
                pred = pred.permute(0, 3, 1, 2)

        # Resize pred to GT spatial resolution if needed
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)

        # Build a spatial valid mask from output['valid_bev'] collapsed over channels
        vb = output.get("valid_bev")
        if vb is None:
            # fallback: everything valid
            vb_spatial = torch.ones(gt.shape[0], *gt.shape[2:], dtype=torch.bool, device=gt.device)
        else:
            # vb may be NCHW (N,C,H,W), NHWC (N,H,W,C) or (N,H,W).
            if vb.ndim == 4:
                # Try candidate interpretations and pick the one whose spatial
                # shape matches GT (after optional resize) or is closest.
                cand1 = (vb.sum(dim=1) > 0)  # assume NCHW -> (N,H,W)
                cand2 = (vb.sum(dim=-1) > 0)  # assume NHWC -> (N,H,W)

                # Prefer candidate that already matches GT spatial dims
                if cand1.shape[1:] == gt.shape[2:]:
                    vb_spatial = cand1
                elif cand2.shape[1:] == gt.shape[2:]:
                    vb_spatial = cand2
                else:
                    # pick cand1 by default; will be resized below
                    vb_spatial = cand1
            elif vb.ndim == 3:
                vb_spatial = vb > 0
            else:
                vb_spatial = vb.bool()

            # Resize vb_spatial to GT size if needed
            if vb_spatial.shape[1:] != gt.shape[2:]:
                vb_spatial = F.interpolate(vb_spatial.unsqueeze(1).float(), size=gt.shape[2:], mode='nearest').squeeze(1).bool()

        # Get confidence_map if present and resize to GT size
        conf = data.get("confidence_map")
        if conf is not None:
            # Normalize conf to (N,H,W)
            if conf.ndim == 4 and conf.shape[1] == 1:
                conf = conf.squeeze(1)
            elif conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf.squeeze(-1)

            if conf.shape[1:] != gt.shape[2:]:
                conf = F.interpolate(conf.unsqueeze(1).float(), size=gt.shape[2:], mode='nearest').squeeze(1).long()

        if conf is not None:
            observable_mask = torch.logical_and(vb_spatial, conf == 0)
            non_observable_mask = torch.logical_and(vb_spatial, conf == 1)
        else:
            observable_mask = vb_spatial
            non_observable_mask = torch.logical_not(vb_spatial)

        # Loop per class and accumulate stats
        for class_idx in range(self.num_classes):
            pred_mask = pred[:, class_idx] > 0.5
            gt_mask = gt[:, class_idx] > 0.5

            # For observable areas
            gt_mask_bool = gt_mask.bool()
            tp_observable = torch.logical_and(torch.logical_and(pred_mask, gt_mask_bool), observable_mask).sum()
            fn_observable = torch.logical_and(torch.logical_and(gt_mask_bool, ~pred_mask), observable_mask).sum()
            fp_observable = torch.logical_and(torch.logical_and(~gt_mask_bool, pred_mask), observable_mask).sum()

            # For non-observable areas
            tp_non_observable = torch.logical_and(torch.logical_and(pred_mask, gt_mask_bool), non_observable_mask).sum()
            fn_non_observable = torch.logical_and(torch.logical_and(gt_mask_bool, ~pred_mask), non_observable_mask).sum()
            fp_non_observable = torch.logical_and(torch.logical_and(~gt_mask_bool, pred_mask), non_observable_mask).sum()
            
            # Update the state
            self.tp_observable[class_idx] += tp_observable
            self.fn_observable[class_idx] += fn_observable
            self.fp_observable[class_idx] += fp_observable
            self.tp_non_observable[class_idx] += tp_non_observable
            self.fn_non_observable[class_idx] += fn_non_observable
            self.fp_non_observable[class_idx] += fp_non_observable

    def compute(self):
        raise NotImplemented


class ObservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        # return (self.intersection_observable / (self.union_observable + 1e-6))[self.class_idx]
        intersection_observable = self.tp_observable[self.class_idx]
        union_observable = self.tp_observable[self.class_idx] + self.fn_observable[self.class_idx] + self.fp_observable[self.class_idx]
        return intersection_observable / (union_observable + 1e-6)
    
class UnobservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        # return (self.intersection_non_observable / (self.union_non_observable + 1e-6))[self.class_idx]
        intersection_non_observable = self.tp_non_observable[self.class_idx]
        union_non_observable = self.tp_non_observable[self.class_idx] + self.fn_non_observable[self.class_idx] + self.fp_non_observable[self.class_idx]
        return intersection_non_observable / (union_non_observable + 1e-6)
    
class MeanObservableIOU(IOU):
    def compute(self):
        # return self.intersection_observable.sum() / (self.union_observable.sum() + 1e-6)
        intersection_observable = self.tp_observable.sum()
        union_observable = self.tp_observable.sum() + self.fn_observable.sum() + self.fp_observable.sum()
        return intersection_observable / (union_observable + 1e-6)

class MeanUnobservableIOU(IOU):
    def compute(self):
        # return self.intersection_non_observable.sum() / (self.union_non_observable.sum() + 1e-6)
        intersection_non_observable = self.tp_non_observable.sum()
        union_non_observable = self.tp_non_observable.sum() + self.fn_non_observable.sum() + self.fp_non_observable.sum()
        return intersection_non_observable / (union_non_observable + 1e-6)
    
class mAP(torchmetrics.classification.MultilabelPrecision):
    def __init__(self, num_labels, **kwargs):
        super().__init__(num_labels=num_labels, **kwargs)

    def update(self, output, data):

        # Build observable mask similarly to IOU.update
        pred = output['output']

        # Ensure pred is NCHW and resize to match seg_masks spatially
        gt = data['seg_masks']
        # If gt is NHWC (last dim == channel count), convert to NCHW for size info
        if gt.ndim == 4 and gt.shape[-1] == pred.shape[1]:
            gt_nchw = gt.permute(0, 3, 1, 2)
        else:
            gt_nchw = gt if gt.ndim == 4 else gt.unsqueeze(0)

        if pred.ndim == 4 and pred.shape[1] != gt_nchw.shape[1]:
            if pred.shape[-1] == gt_nchw.shape[1]:
                pred = pred.permute(0, 3, 1, 2)

        if pred.shape[2:] != gt_nchw.shape[2:]:
            pred = F.interpolate(pred, size=gt_nchw.shape[2:], mode='bilinear', align_corners=False)

        # compute vb_spatial same as IOU
        vb = output.get("valid_bev")
        if vb is None:
            vb_spatial = torch.ones(gt_nchw.shape[0], *gt_nchw.shape[2:], dtype=torch.bool, device=gt_nchw.device)
        else:
            if vb.ndim == 4:
                cand1 = (vb.sum(dim=1) > 0)
                cand2 = (vb.sum(dim=-1) > 0)
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

            if vb_spatial.shape[1:] != gt_nchw.shape[2:]:
                vb_spatial = F.interpolate(vb_spatial.unsqueeze(1).float(), size=gt_nchw.shape[2:], mode='nearest').squeeze(1).bool()

        conf = data.get("confidence_map")
        if conf is not None:
            if conf.ndim == 4 and conf.shape[1] == 1:
                conf = conf.squeeze(1)
            elif conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf.squeeze(-1)

            if conf.shape[1:] != gt_nchw.shape[2:]:
                conf = F.interpolate(conf.unsqueeze(1).float(), size=gt_nchw.shape[2:], mode='nearest').squeeze(1).long()

            observable_mask = torch.logical_and(vb_spatial, conf == 0)
        else:
            observable_mask = vb_spatial

        # Now select predicted and target pixels for observed locations
        # pred: N,C,H,W -> permute to N,H,W,C
        pred_nhwc = pred.permute(0, 2, 3, 1)

        # normalize target to NHWC
        if gt.ndim == 4 and gt.shape[-1] == pred_nhwc.shape[-1]:
            target_nhwc = gt
        elif gt.ndim == 4 and gt.shape[1] == pred.shape[1]:
            target_nhwc = gt.permute(0, 2, 3, 1)
        else:
            # fallback: convert nchw form
            target_nhwc = gt.permute(0, 2, 3, 1) if gt.ndim == 4 else gt

        # Select observed pixels per-sample to avoid shape/order mismatches
        pred_list = []
        target_list = []
        batch_size = pred_nhwc.shape[0]
        for i in range(batch_size):
            mask_i = observable_mask[i]
            # Reduce any extra trailing dimensions (e.g., channel) to HxW
            if mask_i.ndim > 2:
                mask_i = mask_i.any(dim=-1)
            # Ensure boolean
            mask_i = mask_i.bool()
            coords = mask_i.nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue
            # coords is (K, 2): (row, col)
            pi = pred_nhwc[i, coords[:, 0], coords[:, 1]]
            ti = target_nhwc[i, coords[:, 0], coords[:, 1]]
            pred_list.append(pi)
            target_list.append(ti)

        if len(pred_list) == 0:
            return

        pred_sel = torch.cat(pred_list, dim=0)
        target_sel = torch.cat(target_list, dim=0)

        super(mAP, self).update(pred_sel, target_sel)
