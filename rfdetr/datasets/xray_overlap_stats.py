"""xray 파노라마 치아 GT의 instance 간 중첩(overlap) 통계.

파노라마에서는 인접 치아가 영상상 겹쳐 보인다. GT가
  (a) 겹친 영역을 양쪽 치아에 모두 포함(중첩 라벨)했는지,
  (b) 경계에서 잘라 서로 맞닿게(contact) 했는지,
  (c) 사이를 띄워(gap) 그렸는지
데이터셋별로 표본 추출해 대략적으로 산출한다.

deploy/predict.py 병합 단계는 bbox 별 마스크를 label map에 덮어쓰는(overwrite) 구조이므로,
중첩 라벨로 학습된 모델은 겹친 영역을 나중에 그려진 치아가 가져간다.

데이터셋
  ① Kaggle (supervisely ann/*.json)  ② Tufts (ann/*.json, 변환)  → XrayPnoramic 로더로 파일 목록 수집
  ③ 자체 데이터 (labelme → COCO annotations.json)

실행 (repo 루트):
  python -m rfdetr.datasets.xray_overlap_stats --num_samples 100 --out_dir outputs/xray_overlap_stats
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image

from trainer import vtk_utils
from trainer.image_utils import cv2_imread, cv2_imwrite

KAGGLE_ROOT = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays'
DEFAULT_DATASETS = {
    'kaggle': f'{KAGGLE_ROOT}/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
    'tufts': f'{KAGGLE_ROOT}/kaggle_2222',
    'own': (f'{KAGGLE_ROOT}/cbct_ios_dcm_latest_0824', f'{KAGGLE_ROOT}/cbct_ios_dcm_latest_0824/annotations.json'),
    # ② Tufts 변환 전 원본 (Labelbox polygon). 변환(rfdetr/util/dataloader.py::tufts_to_coco)에서 중첩이 사라졌는지 비교용
    'tufts_raw': ('tufts_raw', f'{KAGGLE_ROOT}/kaggle_2222/Segmentation/teeth_polygon.pkl', f'{KAGGLE_ROOT}/kaggle_2222/Radiographs'),
}

# rfdetr/util/dataloader.py::universal_to_permanent 와 동일 (유치 A~T -> 영구치 universal 번호).
# dataloader.py는 import 시 tufts_to_coco()가 실행되므로 import하지 않고 복사해 둔다.
TUFTS_PRIMARY_TO_UNIVERSAL = {
    'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
}

# sample = dict(name, img_file, size=(H, W), instances=[(universal_label, [poly(N,2), ...]), ...])
Sample = Dict


def load_supervisely_samples(img_folder: str) -> List[Sample]:
    """①② : 학습 로더(XrayPnoramic)와 같은 규칙으로 image/ann 파일 목록을 모은다 (split 전체)."""
    from rfdetr.datasets.xraypanoramic import XrayPnoramic

    loader = XrayPnoramic(name='all', splits={'all': (0, 1)}, path_lists=[img_folder])
    samples = []
    for img_file, ann_file in zip(loader.source_files, loader.gt_files):
        samples.append(dict(img_file=img_file, ann_file=ann_file))
    return samples


def read_supervisely(sample: Sample) -> Sample:
    with open(sample['ann_file'], 'r') as f:
        data = json.load(f)
    size = data.get('size')
    if size is not None:
        hw = (int(size['height']), int(size['width']))
    else:
        with Image.open(sample['img_file']) as im:
            hw = (im.size[1], im.size[0])
    instances = []
    for obj in data['objects']:
        if not str(obj['classTitle']).isdigit():
            continue
        poly = np.asarray(obj['points']['exterior'], dtype=np.float64).reshape(-1, 2)
        instances.append((int(obj['classTitle']), [poly]))
    return dict(sample, size=hw, instances=instances)


def load_coco_samples(img_folder: str, annot_file: str) -> List[Sample]:
    from pycocotools.coco import COCO

    coco = COCO(annot_file)
    samples = []
    for image_id in coco.getImgIds():
        info = coco.loadImgs([image_id])[0]
        instances = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
            if not isinstance(a.get('segmentation'), list):
                continue  # RLE(crowd)는 없음 (labelme 변환은 polygon만)
            polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in a['segmentation'] if len(p) >= 6]
            if polys:
                instances.append((int(a['category_id']), polys))
        samples.append(dict(
            img_file=os.path.join(img_folder, info['file_name'].replace('\\', '/')),
            size=(int(info['height']), int(info['width'])),
            instances=instances,
        ))
    return samples


def load_tufts_raw_samples(pkl_file: str, img_dir: str) -> List[Sample]:
    """② Tufts 변환 전 원본. object 하나 = 외곽 polygon + 잔조각 polygon 여러 개 (조각 전체의 합집합이 치아).

    순서는 load_supervisely_samples('tufts')와 같게 파일명(소문자) 정렬 -> 같은 seed면 같은 이미지가 뽑힌다.
    """
    import pickle

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    samples = []
    for item in sorted(data, key=lambda v: v['External ID'].lower()):
        img_file = os.path.join(img_dir, item['External ID'])
        with Image.open(img_file) as im:
            hw = (im.size[1], im.size[0])
        instances = []
        for obj in item['Label'].get('objects', []):
            title = obj['title']
            label = int(title) if title.isdigit() else TUFTS_PRIMARY_TO_UNIVERSAL.get(title)
            if label is None:
                continue
            polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in obj['polygons']]
            polys = [p for p in polys if len(p) >= 3]
            if polys:
                instances.append((label, polys))
        samples.append(dict(img_file=img_file, size=hw, instances=instances))
    return samples


def load_samples(src) -> List[Sample]:
    if isinstance(src, (tuple, list)) and src[0] == 'tufts_raw':
        return load_tufts_raw_samples(*src[1:])
    if isinstance(src, (tuple, list)):
        return load_coco_samples(*src)
    return load_supervisely_samples(src)


def is_adjacent(a: int, b: int) -> bool:
    """universal 번호 기준 같은 악궁에서 이웃한 치아 (1~16 상악, 17~32 하악)."""
    return abs(a - b) == 1 and (a <= 16) == (b <= 16)


def rasterize(polys: List[np.ndarray], hw: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """polygon을 bbox crop 크기 마스크로 그린다. return (mask, (x0, y0, x1, y1)).

    polygon이 여러 개면 합집합. (fillPoly 한 번에 넘기면 겹친 조각이 구멍처럼 빠질 수 있어서 하나씩 그린다)
    """
    pts = np.concatenate(polys, axis=0)
    h, w = hw
    x0, y0 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [w, h])
    x1, y1 = np.clip(np.ceil(pts.max(axis=0)).astype(int) + 1, 0, [w, h])
    mask = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), dtype=np.uint8)
    if mask.size:
        for p in polys:
            cv2.fillPoly(mask, [np.round(p - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool), (int(x0), int(y0), int(x1), int(y1))


def paste(mask, box, canvas_box):
    """crop mask를 canvas_box 좌표계 canvas로 옮긴다."""
    cx0, cy0, cx1, cy1 = canvas_box
    out = np.zeros((cy1 - cy0, cx1 - cx0), dtype=bool)
    x0, y0, x1, y1 = box
    ix0, iy0, ix1, iy1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0] = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    return out


def analyze_sample(sample: Sample, contact_px: int = 3, overlap_px: float = 5.0):
    """한 이미지의 instance 쌍별 관계를 계산한다.

    겹친 영역의 두께 = 2 * max(distance transform(inter)) 로 판정한다.
    (두 polygon이 경계선을 공유하면 fillPoly가 경계 픽셀을 양쪽에 다 넣어서 1~2px 선이 겹친 것처럼 보이므로,
     면적 비율로는 진짜 중첩과 구분이 안 된다)

    pair 분류 (인접/비인접 공통):
      overlap         : 겹친 두께 >= overlap_px  (영상상 겹친 영역을 양쪽 치아에 모두 라벨)
      shared_boundary : 겹쳤지만 두께 < overlap_px (경계선 공유/polygon 오차 수준)
      contact         : inter == 0 이고 contact_px 이내로 맞닿음 (경계에서 잘라 그림)
      gap             : 그보다 떨어짐
    """
    hw = sample['size']
    inst = []
    for label, polys in sample['instances']:
        m, box = rasterize(polys, hw)
        area = int(m.sum())
        if area > 0:
            inst.append(dict(label=label, mask=m, box=box, area=area))

    kernel = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    pairs = []
    covered_by_other = [np.zeros_like(v['mask']) for v in inst]
    for i in range(len(inst)):
        for j in range(i + 1, len(inst)):
            a, b = inst[i], inst[j]
            adjacent = is_adjacent(a['label'], b['label'])
            # contact 판정을 위해 contact_px 만큼 넓힌 bbox가 겹치는 쌍만 계산
            ax0, ay0, ax1, ay1 = a['box']
            bx0, by0, bx1, by1 = b['box']
            if (min(ax1, bx1) + contact_px <= max(ax0, bx0) or min(ay1, by1) + contact_px <= max(ay0, by0)):
                if adjacent:
                    pairs.append(dict(a=a['label'], b=b['label'], adjacent=True, kind='gap', ratio=0.0, iou=0.0,
                                      inter=0, thickness=0.0))
                continue
            cbox = (max(min(ax0, bx0) - contact_px, 0), max(min(ay0, by0) - contact_px, 0),
                    min(max(ax1, bx1) + contact_px, hw[1]), min(max(ay1, by1) + contact_px, hw[0]))
            ma, mb = paste(a['mask'], a['box'], cbox), paste(b['mask'], b['box'], cbox)
            inter_mask = ma & mb
            inter = int(inter_mask.sum())
            ratio = inter / min(a['area'], b['area'])
            iou = inter / (a['area'] + b['area'] - inter)
            thickness = 0.0
            if inter > 0:
                dist = cv2.distanceTransform(np.pad(inter_mask, 1).astype(np.uint8), cv2.DIST_L2, 3)
                thickness = 2.0 * float(dist.max())
                kind = 'overlap' if thickness >= overlap_px else 'shared_boundary'
                if kind == 'overlap':
                    covered_by_other[i] |= paste(inter_mask, cbox, a['box'])
                    covered_by_other[j] |= paste(inter_mask, cbox, b['box'])
            else:
                touch = cv2.dilate(ma.astype(np.uint8), kernel).astype(bool) & mb
                kind = 'contact' if touch.any() else 'gap'
            if adjacent or kind in ('overlap', 'shared_boundary'):
                pairs.append(dict(a=a['label'], b=b['label'], adjacent=adjacent, kind=kind,
                                  ratio=float(ratio), iou=float(iou), inter=inter, thickness=thickness))

    inst_overlap_frac = [float(c.sum()) / v['area'] for c, v in zip(covered_by_other, inst)]
    return dict(img_file=sample['img_file'], size=hw, num_instances=len(inst),
                pairs=pairs, inst_overlap_frac=inst_overlap_frac)


def summarize(results: List[dict], inst_frac_thr: float = 0.01) -> dict:
    kinds = ['overlap', 'shared_boundary', 'contact', 'gap']
    adj = [p for r in results for p in r['pairs'] if p['adjacent'] and p['a'] != p['b']]
    nonadj_ov = [p for r in results for p in r['pairs'] if not p['adjacent'] and p['kind'] == 'overlap' and p['a'] != p['b']]
    same_label = [p for r in results for p in r['pairs'] if p['a'] == p['b'] and p['kind'] == 'overlap']
    adj_ov_ratio = np.array([p['ratio'] for p in adj if p['kind'] == 'overlap'])
    adj_ov_thick = np.array([p['thickness'] for p in adj if p['kind'] == 'overlap'])
    fracs = np.array([f for r in results for f in r['inst_overlap_frac']])

    def pct(n, d):
        return round(100.0 * n / d, 1) if d else None

    def q(v, p):
        return round(float(np.percentile(v, p)), 4) if len(v) else None

    return {
        'images': len(results),
        'instances': int(sum(r['num_instances'] for r in results)),
        'adjacent_pairs': len(adj),
        'adjacent_pair_kind_pct': {k: pct(sum(p['kind'] == k for p in adj), len(adj)) for k in kinds},
        'adjacent_overlap_ratio(inter/min_area)': {'median': q(adj_ov_ratio, 50), 'p90': q(adj_ov_ratio, 90),
                                                   'max': q(adj_ov_ratio, 100)},
        'adjacent_overlap_thickness_px': {'median': q(adj_ov_thick, 50), 'p90': q(adj_ov_thick, 90),
                                          'max': q(adj_ov_thick, 100)},
        'images_with_adjacent_overlap_pct': pct(
            sum(any(p['adjacent'] and p['kind'] == 'overlap' for p in r['pairs']) for r in results), len(results)),
        'adjacent_overlap_pairs_per_image': round(sum(p['kind'] == 'overlap' for p in adj) / max(len(results), 1), 2),
        f'instances_overlapped(>={inst_frac_thr:.0%} of own area)_pct': pct(int((fracs >= inst_frac_thr).sum()), len(fracs)),
        'instance_overlapped_area_frac': {'mean': round(float(fracs.mean()), 4) if len(fracs) else None,
                                          'p90': q(fracs, 90)},
        'non_adjacent_overlap_pairs(e.g. upper/lower bite)': len(nonadj_ov),
        'same_label_overlap_pairs(duplicate label)': len(same_label),
    }


def save_overlap_visualization(sample: Sample, out_file: str):
    """GT 윤곽선 + 두 개 이상 instance가 겹친 픽셀(빨강)을 원본 이미지에 그린다."""
    hw = sample['size']
    img = cv2_imread(sample['img_file'], flags=cv2.IMREAD_COLOR)
    if img is None:
        return
    count = np.zeros(hw, dtype=np.uint8)
    colors = vtk_utils.get_teeth_color_table(normalize=False)
    from rfdetr.datasets.xraypanoramic import label_to_fdi
    for label, polys in sample['instances']:
        m = np.zeros(hw, dtype=np.uint8)
        pts = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in polys]
        for p in pts:
            cv2.fillPoly(m, [p], 1)
        count += m
        color = tuple(int(c) for c in colors[label_to_fdi(label, 33)][::-1])
        cv2.polylines(img, [max(pts, key=len)], True, color, 2)  # 외곽선만 (Tufts 원본은 잔조각이 많음)
    overlay = img.copy()
    overlay[count >= 2] = (0, 0, 255)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, f'overlap px (red) = {int((count >= 2).sum())}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2_imwrite(out_file, img)


def run(datasets: Dict[str, object], num_samples: int = 100, seed: int = 0, out_dir: str = 'outputs/xray_overlap_stats',
        contact_px: int = 3, overlap_px: float = 5.0, vis_top: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for name, src in datasets.items():
        samples = load_samples(src)
        n_total = len(samples)
        rng = np.random.RandomState(seed)  # 데이터셋마다 같은 seed -> tufts / tufts_raw 는 같은 이미지
        picked = rng.choice(n_total, size=min(num_samples, n_total), replace=False) if num_samples > 0 else np.arange(n_total)
        samples = [samples[i] for i in sorted(picked)]
        samples = [read_supervisely(s) if 'ann_file' in s else s for s in samples]

        results = [analyze_sample(s, contact_px, overlap_px) for s in tqdm.tqdm(samples, desc=name)]
        summary = summarize(results)
        summary['total_images_in_dataset'] = n_total
        report[name] = summary

        # 이미지별 결과 (pair 상세 포함)
        with open(os.path.join(out_dir, f'{name}_pairs.json'), 'w') as f:
            json.dump(results, f, indent=1)

        # 인접 중첩 면적이 큰 이미지 순으로 시각화
        score = [sum(p['inter'] for p in r['pairs'] if p['adjacent'] and p['kind'] == 'overlap') for r in results]
        for rank, k in enumerate(np.argsort(score)[::-1][:vis_top]):
            base = os.path.splitext(os.path.basename(samples[k]['img_file']))[0]
            save_overlap_visualization(samples[k], os.path.join(out_dir, name, f'top{rank:02d}_{base}.jpg'))

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def check_tufts_conversion(num_samples: int = 100, seed: int = 0, iou_warn: float = 0.9, ann_dir: str = None,
                           tufts_dir: str = None):
    """② Tufts 원본(teeth_polygon.pkl) vs 변환본(kaggle_2222/ann/*.json) 비교.

    rfdetr/util/dataloader.py::create_annotation_info 의 변환 단계
      fillPoly(조각 전체) -> MORPH_OPEN(9x9) -> findContours(RETR_EXTERNAL) -> contours[0] 만 저장
    에서 치아가 빠지거나 모양이 바뀌었는지 이미지/치아 단위로 센다.
    """
    tufts_dir = tufts_dir or DEFAULT_DATASETS['tufts']  # kaggle_2222 폴더
    raw = {os.path.basename(s['img_file']).lower(): s for s in load_tufts_raw_samples(
        os.path.join(tufts_dir, 'Segmentation', 'teeth_polygon.pkl'), os.path.join(tufts_dir, 'Radiographs'))}
    if ann_dir is None:
        conv = {os.path.basename(s['img_file']).lower(): s for s in load_supervisely_samples(tufts_dir)}
    else:
        # 시험 변환 폴더 (tufts_to_coco --out_dir) 비교용: {이미지명}.json
        import glob
        conv = {}
        for ann_file in glob.glob(os.path.join(ann_dir, '*.json')):
            name = os.path.basename(ann_file)[:-len('.json')].lower()
            if name in raw:
                conv[name] = dict(img_file=raw[name]['img_file'], ann_file=ann_file)
    missing_images = sorted(set(raw) - set(conv))
    names = sorted(set(raw) & set(conv))
    rng = np.random.RandomState(seed)
    picked = [names[i] for i in sorted(rng.choice(len(names), size=min(num_samples, len(names)), replace=False))] \
        if num_samples > 0 else names

    stats = defaultdict(int)
    ious, area_ratio, worst = [], [], []
    for name in tqdm.tqdm(picked, desc='tufts conversion'):
        r, c = raw[name], read_supervisely(conv[name])
        hw = r['size']
        conv_by_label = defaultdict(list)
        for label, polys in c['instances']:
            conv_by_label[label].append(polys)
        for label, polys in r['instances']:
            stats['raw_teeth'] += 1
            if not conv_by_label.get(label):
                stats['teeth_missing_in_converted'] += 1
                continue
            cpolys = conv_by_label[label].pop(0)
            mr, br = rasterize(polys, hw)
            mc, bc = rasterize(cpolys, hw)
            box = (min(br[0], bc[0]), min(br[1], bc[1]), max(br[2], bc[2]), max(br[3], bc[3]))
            mr, mc = paste(mr, br, box), paste(mc, bc, box)
            inter, union = int((mr & mc).sum()), int((mr | mc).sum())
            iou = inter / max(union, 1)
            ious.append(iou)
            area_ratio.append(mc.sum() / max(mr.sum(), 1))
            if iou < iou_warn:
                stats[f'teeth_iou<{iou_warn}'] += 1
                worst.append((round(iou, 3), name, label))
        stats['teeth_extra_in_converted'] += sum(len(v) for v in conv_by_label.values())

    ious, area_ratio = np.array(ious), np.array(area_ratio)
    report = {
        'raw_images': len(raw), 'converted_images': len(conv), 'images_missing_in_converted': len(missing_images),
        'missing_examples': missing_images[:10], 'checked_images': len(picked), **stats,
        'iou(raw,converted)': {'mean': round(float(ious.mean()), 4), 'p1': round(float(np.percentile(ious, 1)), 4),
                               'min': round(float(ious.min()), 4)},
        'area_ratio(converted/raw)': {'mean': round(float(area_ratio.mean()), 4),
                                      'p1': round(float(np.percentile(area_ratio, 1)), 4)},
        'worst': sorted(worst)[:10],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description='xray GT instance overlap statistics')
    parser.add_argument('--check_tufts_conversion', action='store_true',
                        help='Tufts 원본 polygon vs 변환본(ann/*.json) 비교만 실행')
    parser.add_argument('--ann_dir', type=str, default=None, help='--check_tufts_conversion 비교 대상 폴더 (기본 kaggle_2222/ann)')
    parser.add_argument('--num_samples', type=int, default=100, help='데이터셋별 표본 이미지 수 (<=0 이면 전체)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='outputs/xray_overlap_stats')
    parser.add_argument('--contact_px', type=int, default=3, help='맞닿음(contact) 판정 거리(px, 원본 해상도)')
    parser.add_argument('--overlap_px', type=float, default=5.0, help='겹친 영역 두께(px, 원본 해상도)가 이 값 이상이면 중첩 라벨')
    parser.add_argument('--vis_top', type=int, default=10, help='데이터셋별 중첩 큰 순 시각화 장수')
    parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    args = parser.parse_args()
    if args.check_tufts_conversion:
        check_tufts_conversion(args.num_samples, args.seed, ann_dir=args.ann_dir)
        return
    run({k: DEFAULT_DATASETS[k] for k in args.datasets}, args.num_samples, args.seed, args.out_dir,
        args.contact_px, args.overlap_px, args.vis_top)


if __name__ == '__main__':
    main()
