"""Refresh pixel metrics from saved predictions without rerunning the GPU."""
import csv
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
from tools.compare_own1022 import DATA, OUT, metrics, aggregate


def main():
    cv2.setNumThreads(2)
    coco=COCO(str(DATA/'val/annotations.json'))
    summary=json.loads((OUT/'summary.json').read_text())
    gt_reference={}
    for name in ['pretrained','parent','best']:
        indexed=defaultdict(list)
        for d in json.loads((OUT/f'{name}_predictions.json').read_text()):indexed[d['image_id']].append(d)
        rows=[]
        for idx,im in enumerate(coco.imgs.values()):
            anns=coco.imgToAnns[im['id']]
            gt=np.stack([coco.annToMask(a).astype(bool) for a in anns])
            dd=indexed[im['id']]
            masks=np.stack([mu.decode(d['segmentation']).astype(bool) for d in dd])
            labels=np.array([d['category_id'] for d in dd]);scores=np.array([d['score'] for d in dd])
            boxes=np.array([d['bbox'] for d in dd]);boxes[:,2:]+=boxes[:,:2]
            row=metrics(gt,masks,labels,scores,boxes,anns,im['height'],im['width'])
            row['file_name']=im['file_name'].replace('\\','/');rows.append(row)
            signature=tuple(row[k] for k in ['overlap_gt','pair_gt','boundary_gt','contact_gt'])
            if name=='pretrained':gt_reference[im['id']]=signature
            else:assert signature==gt_reference[im['id']],(name,im['id'],signature,gt_reference[im['id']])
            if idx%20==0:print('METRICS',name,idx+1,flush=True)
        summary['models'][name].update(aggregate(rows))
        (OUT/f'{name}_per_image.json').write_text(json.dumps(rows,indent=2))
        with (OUT/f'{name}_per_image.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary['metric_verification']='GT pixel counts identical across models; exact distance thresholds stabilized at 1e-4 pixels'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
