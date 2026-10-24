"""Reproducible paired evaluation on the corrected own-data validation split.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python tools/compare_own1022.py
Uses independent instance masks throughout; never flattens overlapping teeth.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
from rfdetr.models import build_model, PostProcess
from rfdetr.models.segmentation_head import expand_normalized_boxes, paste_masks_in_image
from rfdetr.util.box_ops import box_cxcywh_to_xyxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/compare_own1022'
DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
MODELS = {
    'pretrained': ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
    'parent': ROOT / 'outputs/fine_tune_pa_teeth_roi_padding_fixed/checkpoint_best_total.pth',
    'best': ROOT / 'outputs/fine_tune_own1022/checkpoint_best_total.pth',
    'continue': ROOT / 'outputs/fine_tune_own1022_continue/checkpoint_best_total.pth',
}
THRESHOLD = .3


def rle(mask):
    x = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    x['counts'] = x['counts'].decode()
    return x


def edge(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def dilate(mask, radius):
    if radius >= 4:
        x,y,w,h=cv2.boundingRect(mask.astype(np.uint8))
        result=np.zeros_like(mask)
        if w==0: return result
        x0,y0=max(0,x-radius-1),max(0,y-radius-1)
        x1,y1=min(mask.shape[1],x+w+radius+1),min(mask.shape[0],y+h+radius+1)
        result[y0:y1,x0:x1]=cv2.distanceTransform((~mask[y0:y1,x0:x1]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= radius + 1e-4
        return result
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))).astype(bool)


def ratio(a, b):
    return float(a / b) if b else None


def metrics(gt, pred, labels, scores, boxes, anns, h, w):
    # Class-constrained one-to-one assignment. For a unique GT class choose the
    # highest confidence detection; duplicated GT classes use bbox IoU assignment.
    matched = np.zeros_like(gt)
    detected = 0
    for category in {a['category_id'] for a in anns}:
        gi=[i for i,a in enumerate(anns) if a['category_id']==category]
        candidates=np.flatnonzero((labels==category)&(scores>=THRESHOLD))
        if not len(candidates):continue
        if len(gi)==1:
            matched[gi[0]]=pred[candidates[np.argmax(scores[candidates])]];detected+=1
        else:
            gb=np.array([anns[i]['bbox'] for i in gi]);gb[:,2:]+=gb[:,:2]
            pb=boxes[candidates]
            lo=np.maximum(gb[:,None,:2],pb[None,:,:2]);hi=np.minimum(gb[:,None,2:],pb[None,:,2:])
            intr=np.maximum(hi-lo,0).prod(-1)
            union=(gb[:,2:]-gb[:,:2]).prod(-1)[:,None]+np.maximum(pb[:,2:]-pb[:,:2],0).prod(-1)[None,:]-intr
            rr,cc=linear_sum_assignment(-intr/np.maximum(union,1e-9))
            for r,c in zip(rr,cc):matched[gi[r]]=pred[candidates[c]];detected+=1
    gsum = gt.sum(axis=0)
    psum = pred[scores >= THRESHOLD].sum(axis=0)
    gov, pov = gsum >= 2, psum >= 2
    inter = int((gov & pov).sum())
    union = int((gov | pov).sum())
    # Pairwise preservation also requires the correct tooth identities.
    pair_tp = pair_total = pair_pred = 0
    for i in range(len(gt)):
        for j in range(i):
            a,b=anns[i]['bbox'],anns[j]['bbox']
            x0,y0=max(0,int(max(a[0],b[0]))-1),max(0,int(max(a[1],b[1]))-1)
            x1,y1=min(w,int(min(a[0]+a[2],b[0]+b[2]))+2),min(h,int(min(a[1]+a[3],b[1]+b[3]))+2)
            if x1<=x0 or y1<=y0: continue
            ov = gt[i,y0:y1,x0:x1] & gt[j,y0:y1,x0:x1]
            n = int(ov.sum())
            if n:
                pp = matched[i] & matched[j]
                pair_tp += int((pp[y0:y1,x0:x1] & ov).sum())
                pair_total += n
                pair_pred += int(pp.sum())
    tolerance = max(1, round(.002 * math.hypot(h, w)))
    contact_radius = max(1, round(.005 * math.hypot(h, w)))
    near = np.zeros((h, w), np.uint16)
    for g in gt:
        near += dilate(g, contact_radius)
    contact = near >= 2
    bp = br = npred = ngt = cp = cr = cnp = cng = 0
    dices, ious = [], []
    for g, p in zip(gt, matched):
        intr = int((g & p).sum())
        dices.append(2*intr / max(1, int(g.sum())+int(p.sum())))
        ious.append(intr / max(1, int((g | p).sum())))
        bx,by,bw,bh=cv2.boundingRect((g|p).astype(np.uint8))
        margin=tolerance+2
        x0,y0=max(0,bx-margin),max(0,by-margin)
        x1,y1=min(w,bx+bw+margin),min(h,by+bh+margin)
        ge, pe = edge(g[y0:y1,x0:x1]), edge(p[y0:y1,x0:x1])
        ccontact=contact[y0:y1,x0:x1]
        gd, pd = dilate(ge, tolerance), dilate(pe, tolerance)
        bp += int((pe & gd).sum()); br += int((ge & pd).sum())
        npred += int(pe.sum()); ngt += int(ge.sum())
        cp += int((pe & gd & ccontact).sum()); cr += int((ge & pd & ccontact).sum())
        cnp += int((pe & ccontact).sum()); cng += int((ge & ccontact).sum())
    return dict(image_id=int(anns[0]['image_id']), instances=len(gt), detected=detected,
                mean_dice=float(np.mean(dices)), mean_iou=float(np.mean(ious)),
                overlap_gt=int(gov.sum()), overlap_pred=int(pov.sum()), overlap_tp=inter,
                overlap_union=union, overlap_iou=ratio(inter, union),
                overlap_recall=ratio(inter, int(gov.sum())),
                pair_tp=pair_tp, pair_gt=pair_total, pair_pred=pair_pred,
                pair_recall=ratio(pair_tp, pair_total),
                boundary_precision_tp=bp, boundary_recall_tp=br, boundary_pred=npred, boundary_gt=ngt,
                contact_precision_tp=cp, contact_recall_tp=cr, contact_pred=cnp, contact_gt=cng,
                boundary_tolerance_px=tolerance, contact_radius_px=contact_radius)


def aggregate(rows):
    sums = {k: sum(r[k] for r in rows) for k in ['instances','detected','overlap_gt','overlap_pred','overlap_tp','overlap_union','pair_tp','pair_gt','pair_pred',
        'boundary_precision_tp','boundary_recall_tp','boundary_pred','boundary_gt','contact_precision_tp','contact_recall_tp','contact_pred','contact_gt']}
    sums['instance_dice'] = sum(r['mean_dice']*r['instances'] for r in rows)/sums['instances']
    sums['instance_iou'] = sum(r['mean_iou']*r['instances'] for r in rows)/sums['instances']
    for name, num, den in [('overlap_iou','overlap_tp','overlap_union'),('overlap_recall','overlap_tp','overlap_gt'),('overlap_precision','overlap_tp','overlap_pred'),('pair_recall','pair_tp','pair_gt')]:
        sums[name] = ratio(sums[num], sums[den])
    for prefix in ['boundary','contact']:
        p = ratio(sums[prefix+'_precision_tp'], sums[prefix+'_pred']) or 0
        r = ratio(sums[prefix+'_recall_tp'], sums[prefix+'_gt']) or 0
        sums[prefix+'_precision'], sums[prefix+'_recall'] = p, r
        sums[prefix+'_f1'] = 2*p*r/max(1e-12,p+r)
    if 'nms_overlap_tp' in rows[0]:
        for key in ['nms_overlap_tp','nms_overlap_gt','nms_overlap_pred','nms_overlap_union','nms_detections']:
            sums[key]=sum(r[key] for r in rows)
        for key,num,den in [('nms_overlap_iou','nms_overlap_tp','nms_overlap_union'),('nms_overlap_recall','nms_overlap_tp','nms_overlap_gt'),('nms_overlap_precision','nms_overlap_tp','nms_overlap_pred')]:
            sums[key]=ratio(sums[num],sums[den])
    return sums


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(DATA / 'val/annotations.json'))
    coco.dataset.setdefault('info', {})
    for x in coco.dataset['images']:
        x['file_name'] = x['file_name'].replace('\\','/')
    # Check split leakage by content, independent of renamed file stems.
    split_hashes = {}
    for split in ['train','val']:
        c = json.loads((DATA/split/'annotations.json').read_text())
        split_hashes[split] = {hashlib.sha256((DATA/split/im['file_name'].replace('\\','/')).read_bytes()).hexdigest() for im in c['images']}
    summary = dict(dataset=str(DATA), images=len(coco.imgs), annotations=len(coco.anns),
                   identical_train_val_images=len(split_hashes['train'] & split_hashes['val']),
                   confidence_threshold=THRESHOLD, mask_threshold='logit > 0',
                   input='RGB / 255; letterbox width 704; height (ceil(original_h*640/original_w/16)*16//64+1)*64; cv2 INTER_LINEAR; centered zero padding; map ROI logits directly back to original image',
                   models={})
    for name, path in MODELS.items():
        print('MODEL', name, flush=True)
        cache = OUT / f'{name}_predictions.json'
        rowcache = OUT / f'{name}_per_image.json'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        args = copy.deepcopy(checkpoint['args'])
        args.device = 'cuda:0'
        args.pretrain_weights = str(path)
        if cache.exists() and rowcache.exists():
            detections = json.loads(cache.read_text()); rows=json.loads(rowcache.read_text())
        else:
            model = build_model(args)
            load = model.load_state_dict(checkpoint['model'], strict=True)
            print('STRICT LOAD', load, flush=True)
            model = model.to('cuda:0').eval()
            post = PostProcess(args.num_select, args.segmentation_mode, getattr(args,'segmentation_crop_box_scale',1.15))
            rows, detections = [], []
            for idx, im in enumerate(coco.imgs.values()):
                anns = coco.imgToAnns[im['id']]
                rgb = np.array(Image.open(DATA/'val'/im['file_name']).convert('RGB'))
                h,w = rgb.shape[:2]
                ih = (int(math.ceil(h*640/w/16)*16)//64+1)*64
                scale=min(704/w,ih/h); nw,nh=round(w*scale),round(h*scale)
                ox,oy=(704-nw)//2,(ih-nh)//2
                x=np.zeros((ih,704,3),dtype=np.uint8)
                x[oy:oy+nh,ox:ox+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
                x = torch.from_numpy(x.transpose(2,0,1).copy()).float().div_(255).to('cuda:0')
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    y = model([x])
                    prob=y['pred_logits'][0].sigmoid()
                    scores,indices=prob.flatten().topk(args.num_select)
                    qidx=indices//prob.shape[-1]; labels=indices%prob.shape[-1]
                    b=box_cxcywh_to_xyxy(y['pred_boxes'][0,qidx])
                    mb=box_cxcywh_to_xyxy(expand_normalized_boxes(y['pred_boxes'][0,qidx],getattr(args,'segmentation_crop_box_scale',1.15)))
                    factor=torch.tensor([704,ih,704,ih],device='cuda:0')
                    offset=torch.tensor([ox,oy,ox,oy],device='cuda:0')
                    boxes=(b*factor-offset)/scale
                    maskboxes=(mb*factor-offset)/scale
                    masks=paste_masks_in_image(y['pred_masks'][0,qidx],maskboxes,(h,w))>0
                labels=labels.cpu().numpy(); scores=scores.float().cpu().numpy()
                boxes=boxes.float().cpu().numpy(); masks=masks.cpu().numpy()
                gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
                row = metrics(gt,masks,labels,scores,boxes,anns,h,w)
                row['file_name']=im['file_name']; rows.append(row)
                for lab, score, box, mask in zip(labels,scores,boxes,masks):
                    if lab == 0: continue
                    x0,y0,x1,y1=map(float,box)
                    detections.append(dict(image_id=im['id'], category_id=int(lab),score=float(score),bbox=[x0,y0,x1-x0,y1-y0],segmentation=rle(mask)))
                if idx%10==0: print(name,idx+1,'/',len(coco.imgs),'dice',row['mean_dice'],'overlap',row['overlap_gt'],flush=True)
            cache.write_text(json.dumps(detections))
            rowcache.write_text(json.dumps(rows,indent=2))
            del model
            torch.cuda.empty_cache()
        model_metrics=aggregate(rows)
        # Segmentation results must not include bbox: COCO.loadRes otherwise
        # assigns bbox area instead of mask area.
        for mode in ['bbox','segm']:
            dets = [{k:v for k,v in d.items() if k != ('bbox' if mode=='segm' else 'segmentation')} for d in detections]
            dt = coco.loadRes(dets)
            ev = COCOeval(coco,dt,mode); ev.evaluate(); ev.accumulate();ev.summarize()
            model_metrics[mode+'_AP']=float(ev.stats[0]);model_metrics[mode+'_AP50']=float(ev.stats[1]);model_metrics[mode+'_AP75']=float(ev.stats[2])
        model_metrics['path']=str(path)
        model_metrics['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        model_metrics['epoch_zero_based']=checkpoint.get('epoch')
        model_metrics['segmentation_crop_size']=list(args.segmentation_crop_size)
        summary['models'][name]=model_metrics
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        with (OUT/f'{name}_per_image.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(name, json.dumps(model_metrics), flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
