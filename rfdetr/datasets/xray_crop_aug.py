"""파노라마 xray -> crop 기반 학습 샘플 생성 (치근단 방사선 모사 / two-pass 2차 ROI).

XrayPnoaramicInstance(xray_crop_mode=..., xray_crop_prob=...) 에서 원본 해상도(리사이즈 전)에 적용한다. 기본값은 비활성(None).
여러 mode를 같이 쓰면 샘플마다 확률대로 하나를 고른다 (normalize_crop_modes / pick_crop_mode).

mode
  'periapical' : 치근단 방사선(PA) 모사
      - 같은 악궁에서 x 방향으로 이웃한 치아 3~5개를 고르고 crop
      - 노출 형태: 상악만 / 하악만 / 상하악 동시 (jaw_probs)
      - 가로형(구치부, w/h 약 1.39) / 세로형(전치부) 비율
      - 모서리를 사선(chamfer) 또는 둥글게(round) 잘라 검은 배경 처리 (실제 PA 영상 형태)
      - 출력 크기는 긴 변 periapical_long_side 기준, stride 배수로 맞춤
  'teeth_roi'  : two-pass 정밀 seg의 2차 입력
      - 전체 치아 bbox 합집합 + margin 으로 crop 해서 비관심 영역(두개골, 경추 등)을 버림
      - 학습 시 margin/위치를 흔들어 1차 검출 박스 오차를 모사
      - 추론 시에는 1차 예측 박스로 teeth_roi_box(..., train=False) 를 쓰면 같은 규칙으로 crop 된다

라벨은 universal 번호(1~16 상악, 17~32 하악) 기준.
"""
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

UPPER_MAX = 16  # universal 1~16 상악, 17~32 하악
# 전치부(견치 포함): 상악 6~11, 하악 22~27 -> 세로형 PA가 많다
ANTERIOR = set(range(6, 12)) | set(range(22, 28))

DEFAULT_PERIAPICAL = dict(
    num_teeth=(3, 5),
    jaw_probs=(0.4, 0.4, 0.2),      # (상악만, 하악만, 상하악 동시)
    aspect=1.39,                    # 가로형 w/h (1896x1368). 세로형은 1/aspect
    portrait_prob_anterior=0.7,
    portrait_prob_posterior=0.1,
    x_margin=(0.0, 0.15),           # 선택 치아 폭 대비, 양옆 이웃 치아가 일부 보이게
    apex_margin=(0.05, 0.25),       # 치근 쪽 여유 (치아 높이 대비)
    opposing_margin=(0.0, 0.25),    # 반대 악궁 치관이 살짝 보이게
    corner_style_probs=(0.7, 0.3),  # (chamfer, round)
    corner_ratio=(0.06, 0.14),      # 짧은 변 대비 모서리 크기
    min_visible=0.25,               # crop 안에 보이는 bbox 면적 비율이 이보다 작으면 제외
    long_side=640,
)

DEFAULT_TEETH_ROI = dict(
    margin=(0.03, 0.12),            # 치아 영역 크기 대비 각 변 여유 (학습 시 랜덤)
    eval_margin=0.08,               # 추론/검증 시 고정 margin
    shift=0.03,                     # 1차 박스 오차 모사: 각 변 독립 이동 (치아 영역 크기 대비)
    min_visible=0.5,
)


CROP_MODES = ('periapical', 'teeth_roi')


def normalize_crop_modes(mode=None, prob=1.0) -> Dict[str, float]:
    """xray_crop_mode / xray_crop_prob -> {mode: 샘플당 선택 확률}.

    mode: None | 'periapical' | 'periapical,teeth_roi' | ['periapical', 'teeth_roi'] | {'periapical': 0.3, ...}
    prob: float(모든 mode에 같은 값) 또는 mode 순서와 같은 길이의 list. dict mode면 무시.
    샘플마다 mode 하나를 확률대로 고르고, 남은 확률(1 - 합)은 crop 없이 전체 영상.
    확률 합이 1을 넘으면 비율대로 정규화한다 (예: 둘 다 1.0 -> 0.5 / 0.5, 전체 영상 0%).
    """
    if mode is None or (isinstance(mode, (list, tuple)) and len(mode) == 0):
        return {}
    if isinstance(mode, dict):
        probs = {str(k): float(v) for k, v in mode.items()}
    else:
        assert isinstance(mode, (str, list, tuple)), \
            f"xray_crop_mode must be None | str | list | dict, got {mode!r} (예: 'periapical', ['periapical', 'teeth_roi'])"
        modes = [m.strip() for m in mode.split(',')] if isinstance(mode, str) else [str(m) for m in mode]
        if isinstance(prob, (list, tuple)):
            if len(prob) == 1:
                prob = list(prob) * len(modes)
            assert len(prob) == len(modes), f'xray_crop_prob {prob} must match xray_crop_mode {modes}'
            probs = {m: float(p) for m, p in zip(modes, prob)}
        else:
            probs = {m: float(prob) for m in modes}
    for m, p in probs.items():
        assert m in CROP_MODES, f"invalid xray_crop_mode {m!r} (None | {' | '.join(CROP_MODES)})"
        assert p >= 0, f'xray_crop_prob must be >= 0, got {m}={p}'
    probs = {m: p for m, p in probs.items() if p > 0}
    total = sum(probs.values())
    if total > 1.0:
        probs = {m: p / total for m, p in probs.items()}
    return probs


def split_crop_cfg(cfg: Optional[Dict], modes) -> Dict[str, Dict]:
    """xray_crop_cfg -> mode별 cfg. {'periapical': {...}, 'teeth_roi': {...}} 형태면 mode별로,
    아니면 같은 dict를 모든 mode에 쓴다 (각 mode는 자기 DEFAULT_* 키만 사용)."""
    cfg = cfg or {}
    if any(k in CROP_MODES for k in cfg):
        return {m: dict(cfg.get(m) or {}) for m in modes}
    return {m: dict(cfg) for m in modes}


def pick_crop_mode(mode_probs: Dict[str, float], rng=None) -> Optional[str]:
    """mode_probs에서 샘플 하나의 crop mode를 고른다. 전체 영상이면 None."""
    if not mode_probs:
        return None
    u = (rng or np.random).uniform()
    acc = 0.0
    for m, p in mode_probs.items():
        acc += p
        if u < acc:
            return m
    return None


def _rand(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _clip_box(box, img_hw):
    h, w = img_hw
    x0, y0, x1, y1 = box
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def _fit_aspect(box, aspect, img_hw, rng, top_frac=(0.3, 0.7)):
    """box를 w/h == aspect 가 되도록 짧은 쪽을 늘린다 (이미지 경계 안으로 밀어넣음).

    top_frac: 세로를 늘릴 때 늘어난 높이 중 위쪽에 붙일 비율 범위.
              상악 PA는 치근(위)쪽으로, 하악 PA는 아래쪽으로 늘려야 반대 악궁이 과하게 들어오지 않는다.
    """
    h_img, w_img = img_hw
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw / max(bh, 1e-6) < aspect:
        new_w = min(bh * aspect, w_img)
        extra = new_w - bw
        x0 -= extra * rng.uniform(0.3, 0.7)
        x1 = x0 + new_w
    else:
        new_h = min(bw / aspect, h_img)
        extra = new_h - bh
        y0 -= extra * _rand(rng, top_frac)
        y1 = y0 + new_h
    # 경계 밖이면 안쪽으로 평행 이동
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if x1 > w_img:
        x0, x1 = x0 - (x1 - w_img), float(w_img)
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if y1 > h_img:
        y0, y1 = y0 - (y1 - h_img), float(h_img)
    return _clip_box([x0, y0, x1, y1], img_hw)


def periapical_box(boxes_xyxy: np.ndarray, labels: np.ndarray, img_hw: Tuple[int, int],
                   rng: np.random.RandomState, cfg: Optional[Dict] = None):
    """치근단 방사선처럼 이웃 치아 3~5개 영역을 고른다. return (crop_box xyxy, info dict) 또는 (None, None)."""
    cfg = {**DEFAULT_PERIAPICAL, **(cfg or {})}
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    valid = (labels >= 1) & (labels <= 32)
    upper = np.where(valid & (labels <= UPPER_MAX))[0]
    lower = np.where(valid & (labels > UPPER_MAX))[0]

    jaws = []
    for name, idx, p in zip(('upper', 'lower', 'both'), (upper, lower, upper), cfg['jaw_probs']):
        if len(idx) > 0 and (name != 'both' or len(lower) > 0):
            jaws.append((name, idx, p))
    if not jaws:
        return None, None
    probs = np.array([p for _, _, p in jaws], dtype=np.float64)
    jaw, anchor, _ = jaws[rng.choice(len(jaws), p=probs / probs.sum())]

    # anchor 악궁에서 x 순서로 이웃한 k개
    order = anchor[np.argsort((boxes_xyxy[anchor, 0] + boxes_xyxy[anchor, 2]) / 2)]
    k = int(min(rng.randint(cfg['num_teeth'][0], cfg['num_teeth'][1] + 1), len(order)))
    start = rng.randint(0, len(order) - k + 1)
    selected = order[start:start + k]
    sel_boxes = boxes_xyxy[selected]
    x0, y0 = sel_boxes[:, 0].min(), sel_boxes[:, 1].min()
    x1, y1 = sel_boxes[:, 2].max(), sel_boxes[:, 3].max()

    if jaw == 'both':
        # 같은 x 범위의 하악 치아까지 세로로 포함
        lb = boxes_xyxy[lower]
        overlap = (np.minimum(lb[:, 2], x1) - np.maximum(lb[:, 0], x0)) > 0.3 * (lb[:, 2] - lb[:, 0])
        if overlap.any():
            y0, y1 = min(y0, lb[overlap, 1].min()), max(y1, lb[overlap, 3].max())

    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * _rand(rng, cfg['x_margin'])
    x1 += bw * _rand(rng, cfg['x_margin'])
    apex, opposing = bh * _rand(rng, cfg['apex_margin']), bh * _rand(rng, cfg['opposing_margin'])
    if jaw == 'upper':      # 치근은 위, 반대 악궁(하악 치관)은 아래
        y0, y1 = y0 - apex, y1 + opposing
    elif jaw == 'lower':
        y0, y1 = y0 - opposing, y1 + apex
    else:
        y0, y1 = y0 - apex * 0.5, y1 + apex * 0.5

    anterior = np.mean([labels[i] in ANTERIOR for i in selected]) >= 0.5
    portrait_prob = cfg['portrait_prob_anterior'] if anterior else cfg['portrait_prob_posterior']
    aspect = cfg['aspect'] * rng.uniform(0.92, 1.08)
    if rng.uniform() < portrait_prob:
        aspect = 1.0 / aspect
    top_frac = {'upper': (0.75, 1.0), 'lower': (0.0, 0.25), 'both': (0.35, 0.65)}[jaw]
    box = _fit_aspect(_clip_box([x0, y0, x1, y1], img_hw), aspect, img_hw, rng, top_frac=top_frac)

    style = 'chamfer' if rng.uniform() < cfg['corner_style_probs'][0] else 'round'
    info = dict(jaw=jaw, selected=selected, corner_style=style,
                corner_ratio=_rand(rng, cfg['corner_ratio']), portrait=aspect < 1)
    return box, info


def teeth_roi_box(boxes_xyxy: np.ndarray, img_hw: Tuple[int, int], rng: Optional[np.random.RandomState] = None,
                  train: bool = True, cfg: Optional[Dict] = None):
    """전체 치아 bbox 합집합 + margin. train=True면 margin/변 위치를 흔든다 (1차 검출 오차 모사)."""
    cfg = {**DEFAULT_TEETH_ROI, **(cfg or {})}
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(boxes_xyxy) == 0:
        return None
    x0, y0 = boxes_xyxy[:, 0].min(), boxes_xyxy[:, 1].min()
    x1, y1 = boxes_xyxy[:, 2].max(), boxes_xyxy[:, 3].max()
    bw, bh = x1 - x0, y1 - y0
    if train:
        rng = rng or np.random
        m = [_rand(rng, cfg['margin']) for _ in range(4)]
        s = [rng.uniform(-cfg['shift'], cfg['shift']) for _ in range(4)]
    else:
        m, s = [cfg['eval_margin']] * 4, [0.0] * 4
    box = [x0 - bw * (m[0] + s[0]), y0 - bh * (m[1] + s[1]), x1 + bw * (m[2] + s[2]), y1 + bh * (m[3] + s[3])]
    return _clip_box(box, img_hw)


def corner_mask(h: int, w: int, style: str = 'chamfer', ratio: float = 0.1) -> np.ndarray:
    """PA 영상 외곽 마스크 (1: 영상, 0: 잘린 모서리). style: 'chamfer' | 'round'."""
    mask = np.ones((h, w), dtype=np.uint8)
    r = int(round(min(h, w) * ratio))
    if r < 1:
        return mask
    if style == 'round':
        mask[:] = 0
        cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 1, -1)
        cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 1, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(mask, (cx, cy), r, 1, -1)
    else:
        for tri in (
            [(0, 0), (r, 0), (0, r)],
            [(w - 1, 0), (w - 1 - r, 0), (w - 1, r)],
            [(0, h - 1), (r, h - 1), (0, h - 1 - r)],
            [(w - 1, h - 1), (w - 1 - r, h - 1), (w - 1, h - 1 - r)],
        ):
            cv2.fillPoly(mask, [np.array(tri, dtype=np.int32)], 0)
    return mask


def periapical_target_size(crop_hw: Tuple[int, int], long_side: int = 640, stride: int = 64) -> Tuple[int, int]:
    """긴 변 long_side 기준 비율 유지 후 각 변을 stride 배수로 올림. return (h, w)."""
    h, w = crop_hw
    scale = long_side / max(h, w)
    th = int(np.ceil(h * scale / stride) * stride)
    tw = int(np.ceil(w * scale / stride) * stride)
    return max(th, stride), max(tw, stride)


def visible_fraction(boxes_xyxy: np.ndarray, crop_box: Sequence[float]) -> np.ndarray:
    """각 bbox 면적 중 crop 안에 들어오는 비율."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    x0, y0, x1, y1 = crop_box
    iw = np.clip(np.minimum(boxes_xyxy[:, 2], x1) - np.maximum(boxes_xyxy[:, 0], x0), 0, None)
    ih = np.clip(np.minimum(boxes_xyxy[:, 3], y1) - np.maximum(boxes_xyxy[:, 1], y0), 0, None)
    area = np.clip((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]), 1e-6, None)
    return iw * ih / area


def sample_crop(mode: str, boxes_xyxy: np.ndarray, labels: np.ndarray, img_hw: Tuple[int, int],
                rng: np.random.RandomState, train: bool = True, cfg: Optional[Dict] = None):
    """mode에 맞는 crop 계획. return dict(box=int xyxy, keep=bool[N], corner=(style, ratio) | None, target_hw | None) 또는 None."""
    cfg = cfg or {}
    if mode == 'periapical':
        box, info = periapical_box(boxes_xyxy, labels, img_hw, rng, cfg)
        if box is None:
            return None
        min_visible = {**DEFAULT_PERIAPICAL, **cfg}['min_visible']
        corner = (info['corner_style'], info['corner_ratio'])
    elif mode == 'teeth_roi':
        box = teeth_roi_box(boxes_xyxy, img_hw, rng, train=train, cfg=cfg)
        if box is None:
            return None
        min_visible = {**DEFAULT_TEETH_ROI, **cfg}['min_visible']
        corner, info = None, {}
    else:
        raise ValueError(f"unknown xray crop mode: {mode!r} (None | 'periapical' | 'teeth_roi')")

    box = np.array([np.floor(box[0]), np.floor(box[1]), np.ceil(box[2]), np.ceil(box[3])], dtype=np.int64)
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    keep = visible_fraction(boxes_xyxy, box) >= min_visible
    target_hw = None
    if mode == 'periapical':
        target_hw = periapical_target_size((box[3] - box[1], box[2] - box[0]),
                                           long_side={**DEFAULT_PERIAPICAL, **cfg}['long_side'])
    return dict(box=box, keep=keep, corner=corner, target_hw=target_hw, info=info)


def save_crop_samples(img_folder, coco_directories, mode: str, out_dir: str, num: int = 12, seed: int = 0,
                      num_classes: int = 32, augment_en: bool = True, cfg: Optional[Dict] = None):
    """crop 모드로 만든 학습 샘플(영상 + GT 마스크/bbox)을 저장해 눈으로 확인한다.

    학습과 같은 로더(XrayPnoaramicInstance, name='train')로 읽는다. 공개 데이터(①②)와 자체 데이터(③)를 섞어서 뽑는다.
    저장: {out_dir}/{mode}/{순번:02d}_{pub|own}_{파일명 또는 idx}.jpg  (왼쪽 아래: 데이터, 입력 크기, 치아 수)
    """
    import os
    from rfdetr.datasets.xraypanoramic import XrayPnoaramicInstance, label_to_fdi
    from trainer import vtk_utils
    from trainer.image_utils import cv2_imwrite

    ds = XrayPnoaramicInstance(img_folder, include_masks=True, num_classes=num_classes, name='train',
                               augment_en=augment_en, coco_directories=coco_directories, xray_crop_mode=mode,
                               xray_crop_prob=1.0, xray_crop_cfg=cfg)
    colors = vtk_utils.get_teeth_color_table(normalize=False)
    n_coco = len(ds.coco_apis) if ds.coco_apis is not None else 0
    rng = np.random.RandomState(seed)
    n_own = min(num // 4, n_coco)  # 1/4은 자체 데이터
    picks = list(rng.choice(n_coco, n_own, replace=False)) if n_own else []
    picks += list(n_coco + rng.choice(len(ds.source_files), num - n_own, replace=False))
    np.random.seed(seed)  # crop/증강 랜덤도 고정
    save_dir = os.path.join(out_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    for k, i in enumerate(picks):
        img, t = ds.parse_item_coco(int(i))
        img = (np.transpose(np.asarray(img), (1, 2, 0)) * 255).astype(np.uint8)
        img = np.ascontiguousarray(np.repeat(img, 3, 2) if img.shape[2] == 1 else img)
        h, w = img.shape[:2]
        masks = np.asarray(t['masks']).astype(bool)
        labels = np.asarray(t['labels']).reshape(-1)
        overlay = img.copy()
        for m, lab in zip(masks, labels):
            overlay[m] = colors[label_to_fdi(int(lab), 33)]
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        for (cx, cy, bw, bh), lab in zip(np.asarray(t['boxes']).reshape(-1, 4), labels):
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (255, 255, 0), 1)
            cv2.putText(img, str(label_to_fdi(int(lab), 33)), (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1)
        src = 'own' if i < n_coco else 'pub'
        cv2.putText(img, f'{src} {w}x{h} n={len(labels)}', (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # 자체 데이터 파일명에는 환자 이름이 있어서 index로 저장
        name = f'idx{int(i)}' if src == 'own' else os.path.splitext(os.path.basename(ds.file_name(int(i))))[0]
        cv2_imwrite(os.path.join(save_dir, f'{k:02d}_{src}_{name}.jpg'), img[..., ::-1])
    print(f'saved {len(picks)} {mode} samples to {save_dir}')
    return save_dir
