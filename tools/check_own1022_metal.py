"""Inspect bright prosthesis candidates; brightness is a proxy, not a metal GT."""
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pycocotools.coco import COCO
from pycocotools import mask as mu
from rfdetr.engine import non_max_suppression
from tools.compare_own1022 import DATA,OUT

FONT='/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'


def panel(rgb,gt,pred,bright,title,box,mode):
    x0,y0,x1,y1=box
    source=rgb[y0:y1,x0:x1].copy();g=gt[y0:y1,x0:x1];b=bright[y0:y1,x0:x1]
    if mode=='GT':
        source[b]=(source[b]*.45+np.array([255,220,20])*.55).astype(np.uint8)
    else:
        p=pred[y0:y1,x0:x1]
        source[p]=(source[p]*.65+np.array([30,190,255])*.35).astype(np.uint8)
        miss=g&~p
        source[miss]=(source[miss]*.2+np.array([255,45,45])*.8).astype(np.uint8)
    contours,_=cv2.findContours(g.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(source,contours,-1,(255,235,90),1)
    if mode!='GT':
        contours,_=cv2.findContours(pred[y0:y1,x0:x1].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source,contours,-1,(70,200,255),1)
    image=Image.fromarray(source)
    scale=min(800/image.width,440/image.height)
    image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(800,490),(16,22,30));canvas.paste(image,((800-image.width)//2,50+(440-image.height)//2))
    ImageDraw.Draw(canvas).text((12,12),title,font=ImageFont.truetype(FONT,19),fill='white')
    return canvas


def main():
    cv2.setNumThreads(2)
    coco=COCO(str(DATA/'val/annotations.json'))
    candidates=json.loads((OUT/'metal_candidates.json').read_text())
    indexed={}
    for name in ['pretrained','parent','best']:
        indexed[name]=defaultdict(list)
        for d in json.loads((OUT/f'{name}_predictions.json').read_text()):
            if d['score']>.5:indexed[name][d['image_id']].append(d)
    rows=[];sheets={}
    (OUT/'metal').mkdir(exist_ok=True)
    for item in candidates:
        i=item['image_id'];im=coco.imgs[i];rgb=np.array(Image.open(DATA/'val'/im['file_name'].replace('\\','/')).convert('RGB'))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        gt=np.any([coco.annToMask(a).astype(bool) for a in coco.imgToAnns[i]],axis=0)
        box=item['crop_box'];x0,y0,x1,y1=box
        roi=np.zeros(gt.shape,bool);roi[y0:y1,x0:x1]=True
        bright=(gray>=240)&gt&roi
        row=dict(image_id=i,crop_box=box,bright_gt_pixels=int(bright.sum()),models={})
        panels=[panel(rgb,gt,gt,bright,f'GT | image {i:03d} | yellow = bright GT',box,'GT')]
        for name in ['pretrained','parent','best']:
            dd=indexed[name][i];boxes=np.array([d['bbox'] for d in dd]).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
            keep=non_max_suppression(boxes,np.array([d['score'] for d in dd]),threshold=.4) if len(dd) else []
            pred=np.zeros(gt.shape,bool)
            for j in keep:pred|=mu.decode(dd[j]['segmentation']).astype(bool)
            miss=int((bright&~pred).sum());gtroi=gt&roi
            row['models'][name]=dict(bright_missing_pixels=miss,bright_missing_ratio=miss/max(1,int(bright.sum())),whole_gt_missing_ratio=int((gtroi&~pred).sum())/max(1,int(gtroi.sum())))
            if name!='parent':panels.append(panel(rgb,gt,pred,bright,f'{name} | bright GT missed {miss/max(1,int(bright.sum()))*100:.2f}%',box,'pred'))
        sheet=Image.new('RGB',(2400,490))
        for j,p in enumerate(panels):sheet.paste(p,(800*j,0))
        sheet.save(OUT/'metal'/f'metal_{i:03d}.jpg',quality=95)
        rows.append(row);sheets[i]=sheet
    # Show two strongest underfill examples and two typical examples; all
    # sixteen examined cases remain available in the report.
    ranked=sorted(rows,key=lambda r:r['models']['best']['bright_missing_ratio'],reverse=True)
    ids=[r['image_id'] for r in ranked[:2]+ranked[len(ranked)//2:len(ranked)//2+2]]
    preview=Image.new('RGB',(2400,490*len(ids)))
    for j,i in enumerate(ids):preview.paste(sheets[i],(0,490*j))
    preview.save(OUT/'metal_comparison_preview.jpg',quality=95)
    summary=dict(method='16 visually reviewed bright prosthesis crops; gray >= 240 within GT is a brightness proxy, not metal material GT; score > .5; class-agnostic NMS .4; union coverage, no tooth-label matching',
                 cases=len(rows),preview_image_ids=ids,models={})
    for name in ['pretrained','parent','best']:
        denom=sum(r['bright_gt_pixels'] for r in rows);miss=sum(r['models'][name]['bright_missing_pixels'] for r in rows)
        summary['models'][name]=dict(bright_gt_pixels=denom,bright_missing_pixels=miss,bright_missing_ratio=miss/denom)
    (OUT/'metal_summary.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2))
    figures=''.join(f'<figure><a href="metal/metal_{r["image_id"]:03d}.jpg"><img loading="lazy" src="metal/metal_{r["image_id"]:03d}.jpg"></a><figcaption>Image {r["image_id"]:03d} · best bright missing {r["models"]["best"]["bright_missing_ratio"]*100:.2f}%</figcaption></figure>' for r in ranked)
    section=f'''<h2>금속 보철 후보 영역의 과소 분할 점검</h2><p>밝은 보철 후보 16개 영역을 원본 영상으로 육안 확인했습니다. 전용 금속 재질 라벨은 없으므로 GT 내부 grayscale ≥ 240 픽셀을 밝은 보철 영역의 대용 지표로 사용합니다. 금속 분류 정확도나 전체 금속 부위 성능을 의미하지 않습니다. NMS 후 예측 마스크 합집합의 커버리지이며 치아 번호 일치 여부는 여기서 평가하지 않습니다.</p><p>GT 내부 밝은 영역의 누락 비율: pretrained {summary['models']['pretrained']['bright_missing_ratio']*100:.2f}%, parent {summary['models']['parent']['bright_missing_ratio']*100:.2f}%, best {summary['models']['best']['bright_missing_ratio']*100:.2f}%. 낮을수록 좋습니다. 왼쪽 GT의 노랑은 밝은 평가 영역입니다. 예측 패널의 파랑은 예측 마스크, 빨강은 GT에서 누락된 부분(밝은 영역 외 일반 치아 누락도 포함)입니다. 구멍의 내부 윤곽도 표시합니다.</p>{figures}'''
    report=OUT/'report.html';body=report.read_text()
    body=body.split('<h2>금속 보철 후보 영역의 과소 분할 점검</h2>')[0].replace('</html>','')
    report.write_text(body+section+'</html>')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
