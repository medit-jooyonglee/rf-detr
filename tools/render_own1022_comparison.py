"""Render independent instance contours and overlap error maps from saved RLEs."""
import html
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
import importlib
import tools.compare_own1022 as comparison
aggregate=importlib.reload(comparison).aggregate
from rfdetr.engine import non_max_suppression

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/compare_own1022'
DATA=Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
FONT='/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'


def color(k):
    hsv=np.uint8([[[int(k*137.5)%180,200,255]]])
    return tuple(map(int,cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)[0,0]))


def draw_panel(rgb, masks, labels, title, box=None, target_w=960, target_h=430, error=None, gt_overlap=None):
    h,w=rgb.shape[:2]
    if box is None:box=(0,0,w,h)
    x0,y0,x1,y1=box
    patch=rgb[y0:y1,x0:x1].copy()
    scale=min(target_w/(x1-x0),target_h/(y1-y0))
    nw,nh=max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    patch=cv2.resize(patch,(nw,nh))
    if error is not None:
        po=error[y0:y1,x0:x1];go=gt_overlap[y0:y1,x0:x1]
        map_=np.zeros((y1-y0,x1-x0,3),np.uint8)
        map_[po&go]=(40,235,100)
        map_[go&~po]=(255,65,65)
        map_[po&~go]=(40,140,255)
        map_=cv2.resize(map_,(nw,nh),interpolation=cv2.INTER_NEAREST)
        active=map_.max(axis=2)>0
        patch[active]=(patch[active]*.2+map_[active]*.8).astype(np.uint8)
    else:
        for mask,label in zip(masks,labels):
            crop=cv2.resize(mask[y0:y1,x0:x1].astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)
            contours,_=cv2.findContours(crop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(patch,contours,-1,color(label),2)
    canvas=Image.new('RGB',(target_w,target_h+45),(16,22,30))
    canvas.paste(Image.fromarray(patch),((target_w-nw)//2,45+(target_h-nh)//2))
    ImageDraw.Draw(canvas).text((12,10),title,font=ImageFont.truetype(FONT,20),fill='white')
    return canvas


def main():
    cv2.setNumThreads(2)
    summary=json.loads((OUT/'summary.json').read_text())
    logs=[json.loads(x) for x in (ROOT/'outputs/fine_tune_own1022/log.txt').read_text().splitlines() if x.strip()]
    fig,ax=plt.subplots(figsize=(10,3.5))
    ax.plot([d['epoch']+1 for d in logs],[100*d['test_coco_eval_masks'][0] for d in logs],label='Regular validation mask AP')
    ax.plot([d['epoch']+1 for d in logs],[100*d['ema_test_coco_eval_masks'][0] for d in logs],label='EMA validation mask AP')
    ax.axvline(15,color='gray',ls='--',label='Saved best (epoch 15)')
    ax.set(xlabel='Epoch',ylabel='COCO mask AP (%)',title='Completed 100 epochs; mixed public + own validation data')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'training_curve.png',dpi=160);plt.close(fig)
    coco=COCO(str(DATA/'val/annotations.json'))
    models=['pretrained','parent','best']
    predictions={};rows={}
    for name in models:
        indexed=defaultdict(list)
        for d in json.loads((OUT/f'{name}_predictions.json').read_text()):
            if d['score']>=.3:indexed[d['image_id']].append(d)
        predictions[name]=indexed
        rows[name]={r['image_id']:r for r in json.loads((OUT/f'{name}_per_image.json').read_text())}
    # Supplement raw COCO detections with the repository's class-agnostic NMS,
    # so duplicate predictions cannot masquerade as anatomical overlap.
    for i,im in coco.imgs.items():
        gt=np.stack([coco.annToMask(a).astype(bool) for a in coco.imgToAnns[i]])
        go=gt.sum(axis=0)>=2
        for name in models:
            dd=[d for d in predictions[name][i] if d['score']>.5]
            pb=np.array([d['bbox'] for d in dd],dtype=float).reshape(-1,4)
            pb[:,2:]+=pb[:,:2]
            keep=non_max_suppression(pb,np.array([d['score'] for d in dd]),threshold=.4) if len(dd) else []
            count=np.zeros(go.shape,np.uint16)
            for j in keep:count+=mu.decode(dd[j]['segmentation']).astype(bool)
            po=count>=2
            rows[name][i].update(nms_overlap_tp=int((go&po).sum()),nms_overlap_gt=int(go.sum()),nms_overlap_pred=int(po.sum()),nms_overlap_union=int((go|po).sum()),nms_detections=len(keep))
        if i%20==0:print('NMS',i,flush=True)
    for name in models:
        rr=list(rows[name].values())
        (OUT/f'{name}_per_image.json').write_text(json.dumps(rr,indent=2))
        with (OUT/f'{name}_per_image.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
        summary['models'][name].update(aggregate(rr))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    audit=json.loads((OUT/'data_audit.json').read_text())
    excluded={x['val_image_id'] for x in audit['own1022_identical_train_val_images']}
    ids=[i for i in coco.imgs if i not in excluded]
    clean=dict(images=len(ids),excluded_image_ids=sorted(excluded),models={})
    coco.dataset.setdefault('info',{})
    for name in models:
        clean['models'][name]=aggregate([rows[name][i] for i in ids])
        for mode in ['bbox','segm']:
            detections=json.loads((OUT/f'{name}_predictions.json').read_text())
            dets=[{k:v for k,v in d.items() if k!=('bbox' if mode=='segm' else 'segmentation')} for d in detections if d['image_id'] in ids]
            dt=coco.loadRes(dets);ev=COCOeval(coco,dt,mode);ev.params.imgIds=ids
            ev.evaluate();ev.accumulate();ev.summarize()
            for j,k in enumerate(['AP','AP50','AP75']):clean['models'][name][mode+'_'+k]=float(ev.stats[j])
    (OUT/'clean_summary.json').write_text(json.dumps(clean,indent=2))
    summary_all=summary
    summary=clean
    # Paired image bootstrap: fixed seed, 2000 resamples; accounts for pixels
    # within the same image, does not claim patient-level independence.
    rng=np.random.default_rng(42)
    boot={}
    for ref in ['pretrained','parent']:
        deltas={k:[] for k in ['dice','overlap_iou','pair_recall','contact_f1']}
        for _ in range(2000):
            sample=rng.choice(ids,len(ids),replace=True)
            for k in deltas:
                vals=[]
                for name in [ref,'best']:
                    rr=[rows[name][int(i)] for i in sample]
                    if k=='dice':v=sum(r['mean_dice']*r['instances'] for r in rr)/sum(r['instances'] for r in rr)
                    elif k=='overlap_iou':v=sum(r['overlap_tp'] for r in rr)/max(1,sum(r['overlap_union'] for r in rr))
                    elif k=='pair_recall':v=sum(r['pair_tp'] for r in rr)/max(1,sum(r['pair_gt'] for r in rr))
                    else:
                        p=sum(r['contact_precision_tp'] for r in rr)/max(1,sum(r['contact_pred'] for r in rr))
                        r=sum(r['contact_recall_tp'] for r in rr)/max(1,sum(r['contact_gt'] for r in rr))
                        v=2*p*r/max(1e-12,p+r)
                    vals.append(v)
                deltas[k].append(vals[1]-vals[0])
        boot[ref]={k:list(np.percentile(v,[2.5,97.5])) for k,v in deltas.items()}
    (OUT/'paired_bootstrap.json').write_text(json.dumps(boot,indent=2))
    eligible=[i for i in ids if rows['best'][i]['overlap_gt']>0]
    ranked=sorted(eligible,key=lambda i:(rows['best'][i]['overlap_iou'] or 0)-(rows['pretrained'][i]['overlap_iou'] or 0))
    selected=list(dict.fromkeys(ranked[-3:][::-1]+ranked[:3]+ranked[len(ranked)//2-1:len(ranked)//2+2]))
    (OUT/'images').mkdir(exist_ok=True)
    figures=[];thumbs=[]
    for index,i in enumerate(selected):
        im=coco.imgs[i]; anns=coco.imgToAnns[i]
        rgb=np.array(Image.open(DATA/'val'/im['file_name'].replace('\\','/')).convert('RGB'))
        gt=np.stack([coco.annToMask(a).astype(bool) for a in anns]);gl=[a['category_id'] for a in anns]
        go=gt.sum(axis=0)>=2
        masks={name:[mu.decode(d['segmentation']).astype(bool) for d in predictions[name][i]] for name in models}
        labels={name:[d['category_id'] for d in predictions[name][i]] for name in models}
        ov={name:np.sum(masks[name],axis=0)>=2 for name in models}
        # Select a region from GT overlap alone, so both models see the same
        # crop without selecting only one model's most favorable pixels.
        n,cc,stats,centers=cv2.connectedComponentsWithStats(go.astype(np.uint8))
        if n>1:
            component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);cx,cy=centers[component]
        else:cx,cy=rgb.shape[1]/2,rgb.shape[0]/2
        width=min(rgb.shape[1],round(rgb.shape[1]*.27));height=min(rgb.shape[0],round(width*.6))
        x0=max(0,min(rgb.shape[1]-width,round(cx-width/2)));y0=max(0,min(rgb.shape[0]-height,round(cy-height/2)))
        box=(x0,y0,x0+width,y0+height)
        sheet=Image.new('RGB',(4*800,3*(380+45)),(16,22,30))
        collections=[('GT',gt,gl)]+[(name,masks[name],labels[name]) for name in models]
        for col,(name,mm,ll) in enumerate(collections):
            title=f'Corrected GT | image {i:03d}' if name=='GT' else f'{name} | Dice {rows[name][i]["mean_dice"]:.3f} | overlap IoU {(rows[name][i]["overlap_iou"] or 0):.3f}'
            panel=draw_panel(rgb,mm,ll,title,target_w=800,target_h=380)
            sheet.paste(panel,(col*800,0))
            panel=draw_panel(rgb,mm,ll,name+' | same overlap crop',box,800,380)
            sheet.paste(panel,(col*800,425))
            panel=draw_panel(rgb,mm,ll,'GT overlap' if name=='GT' else name+' | green=TP red=miss blue=extra',box,800,380,error=go if name=='GT' else ov[name],gt_overlap=go)
            sheet.paste(panel,(col*800,850))
        fname=f'comparison_{i:03d}.jpg';sheet.save(OUT/'images'/fname,quality=95)
        delta=(rows['best'][i]['overlap_iou'] or 0)-(rows['pretrained'][i]['overlap_iou'] or 0)
        tag='improved' if index<3 else 'regressed' if index<6 else 'typical'
        figures.append(f'<figure><a href="images/{fname}"><img loading="lazy" src="images/{fname}"></a><figcaption>Image {i:03d} · {tag} · overlap IoU Δ {delta*100:+.2f} pp · full / same GT-selected crop / overlap error</figcaption></figure>')
        if index in [0,3,6]:
            # Compact, GT/pretrained/best zoom for the chat preview.
            thumb=Image.new('RGB',(3*700,400))
            for col,name in enumerate(['GT','pretrained','best']):
                mm,ll=(gt,gl) if name=='GT' else (masks[name],labels[name])
                thumb.paste(draw_panel(rgb,mm,ll,f'{name} | image {i:03d}',box,700,355),(col*700,0))
            thumbs.append(thumb)
    montage=Image.new('RGB',(2100,400*len(thumbs)))
    for j,t in enumerate(thumbs):montage.paste(t,(0,j*400))
    montage.save(OUT/'comparison_preview.jpg',quality=95)
    columns=['segm_AP','segm_AP50','segm_AP75','bbox_AP','instance_dice','instance_iou','overlap_iou','overlap_recall','overlap_precision','pair_recall','nms_overlap_iou','nms_overlap_recall','nms_overlap_precision','boundary_f1','contact_f1']
    table='<table><tr><th>지표 (%)</th><th>pretrained</th><th>학습 시작점 parent</th><th>own1022 best</th><th>best − pretrained (%p)</th><th>best − parent (%p)</th></tr>'
    for k in columns:
        values=[summary['models'][m][k] for m in models]
        table+='<tr><td>'+k+'</td>'+''.join(f'<td>{v*100:.2f}</td>' for v in values)+f'<td>{(values[2]-values[0])*100:+.2f}</td><td>{(values[2]-values[1])*100:+.2f}</td></tr>'
    table+='</table>'
    improved=sum((rows['best'][i]['overlap_iou'] or 0)>(rows['pretrained'][i]['overlap_iou'] or 0) for i in eligible)
    stats=dict(overlap_images=len(eligible),overlap_iou_improved_images=improved,selected_image_ids=selected,
               bootstrap_95_ci_delta=boot)
    (OUT/'visual_summary.json').write_text(json.dumps(stats,indent=2))
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>own1022 모델 비교</title>
    <style>body{{font-family:system-ui,sans-serif;background:#10161e;color:#e2e8f0;max-width:1600px;margin:30px auto;padding:24px;line-height:1.6}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{border:1px solid #374151;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}figure{{margin:32px 0}}a{{color:#71baff}}code{{color:#7eeac5}}.note{{padding:20px;background:#1d2939;border-radius:12px}}</style>
    <h1>보완 라벨에 대한 pretrained / parent / own1022 best 비교</h1>
    <p>2026-10-23 · own-data val 85장, 치아 2,319개 · 같은 영상, 전처리, 원본 해상도 GT · GPU 1, FP32</p>
    <div class="note">학습은 100 epoch를 완료했습니다. checkpoint_best_total은 0-based epoch 14의 regular 모델과 일치합니다.
    best는 전체 공개+자체 검증 데이터의 마스크 AP로 선택됐습니다. 마지막 epoch와는 다릅니다.
    pretrained → parent(이전 crop/padding 학습) → own1022 순서이므로, best−parent가 이번 학습에 더 가까운 비교입니다.
    검증 85장 모두 구버전 0824 데이터의 영상과 SHA256이 일치합니다. 별도 <code>data_audit.json</code>에서 확인할 수 있습니다.
    따라서 이 결과는 보완 라벨에 대한 적합도 비교이며, 미관측 환자 일반화 또는 라벨 보완 단독의 인과효과를 입증하지 않습니다.</div>
    <h2>학습 상태</h2><img src="training_curve.png" alt="100 epoch validation curves">
    <h2>수치 비교 — train과 중복된 3장을 제외한 82장</h2>{table}
    <p>val image ID 0, 6, 60은 train과 SHA256이 같아 이 표와 아래 bootstrap/사례 선정에서 제외했습니다. 전체 85장 수치는 <a href="summary.json">summary.json</a>, 제외 후 수치는 <a href="clean_summary.json">clean_summary.json</a>에 있습니다. 이 제외는 이번 학습의 직접 중복을 줄이는 조치이며, pretrained의 구버전 학습 노출은 제거하지 못합니다.</p>
    <p>GT 중첩이 있는 {len(eligible)}장 중 overlap IoU가 개선된 영상: {improved}장. 전체 수치는 픽셀/인스턴스를 합산하므로 영상별 단순평균과 다릅니다.</p>
    <h2>측정 방법</h2><ul>
    <li>학습 로더의 비율 유지 letterbox (기준 640, stride 64 → 실제 너비 704), RGB/255, 검은 중앙 패딩. ROI 마스크 logits를 패딩 좌표에서 원본 영상 좌표로 직접 복원합니다. 원본 COCO polygons를 GT로 사용합니다.</li>
    <li>COCO AP는 상위 35개 검출과 클래스 1~32를 사용합니다. 다른 지표와 그림은 score ≥ 0.3, mask logit &gt; 0. 클래스별 최고 점수 예측을 해당 GT 치아와 대응하며 누락은 Dice/IoU 0으로 계산합니다. 5개 영상에 중복 치아번호가 있어 해당 클래스는 bbox IoU 기반 Hungarian 일대일 대응을 적용합니다. 중복 번호 목록은 data_audit.json에 있습니다.</li>
    <li>overlap은 두 개 이상의 독립 치아 마스크가 겹친 픽셀입니다. overlap IoU / recall / precision은 실제 중첩 영역과 예측 중첩 영역의 교집합으로 계산합니다. pair_recall은 올바른 두 치아 번호가 GT 겹침을 함께 덮는 비율입니다.</li>
    <li>기본 지표와 그림은 COCO 상위 35개 원시 검출입니다. nms_overlap_*는 추가로 기존 코드와 같은 confidence &gt; 0.5, 클래스 무관 bbox NMS IoU 0.4를 적용해 중복 예측을 제거한 결과입니다. FDI 번호 강제 재배열은 적용하지 않습니다. 실제 추론 파이프라인 전체와 동일하다고 주장하지 않습니다.</li>
    <li>boundary F1은 치아별 경계를 허용 오차(영상 대각선의 0.2%) 안에서 비교합니다. contact F1은 GT 치아를 대각선 0.5%만큼 팽창시켜 두 치아가 가까운 구역으로 제한합니다. 중첩을 무조건 늘리는 모델은 precision에서 불이익을 받습니다.</li>
    <li>95% 구간은 영상 단위 paired bootstrap 2,000회, seed 42. 환자 단위 독립성은 보장하지 않습니다. 결과: <a href="paired_bootstrap.json">paired_bootstrap.json</a>.</li>
    <li>학습 args의 freeze_encoder=False이며 backbone, transformer, coarse/refinement head 가중치가 모두 변경됐습니다. 현재 저장된 설정과 가중치는 모델 일부를 고정한 학습으로 보이지 않습니다.</li>
    </ul><h2>같은 영상 비교</h2><p>왼쪽부터 보완 GT / pretrained / parent / best. 위부터 전체 영상 / GT 중첩 기준 동일 확대 / 중첩 오류 지도.
    초록=보존, 빨강=누락, 파랑=추가 중첩. 개선·악화·중앙값 사례를 함께 제시합니다. 클릭하면 원본 크기로 열립니다. 치아별 색은 세 모델에서 동일하며, 단일 라벨맵으로 합치지 않고 각각의 윤곽을 그립니다.</p>
    {''.join(figures)}
    <h2>재현 및 원자료</h2><p><code>CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python tools/compare_own1022.py</code><br>
    <code>PYTHONPATH=. /opt/conda/bin/python tools/render_own1022_comparison.py</code></p>
    <p><a href="summary.json">전체 지표 / 체크포인트 SHA256</a> · <a href="data_audit.json">데이터 중복 감사</a> · <a href="pretrained_per_image.csv">pretrained 영상별 CSV</a> · <a href="parent_per_image.csv">parent 영상별 CSV</a> · <a href="best_per_image.csv">best 영상별 CSV</a> · <a href="evaluation.log">실행 로그</a></p></html>'''
    (OUT/'report.html').write_text(body)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
