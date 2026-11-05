"""Quick qualitative check of the still-training outputs/peri_data_fusion run.

checkpoint_best_ema.pth hasn't moved past epoch 106 (its val-metric apparently
hasn't improved since), even though training has continued well past that
(outputs/peri_data_fusion/checkpoint.pth is already much further along). To
see what the *current* training state actually produces, this pulls the EMA
weights bundled in the live checkpoint.pth (checkpoint['ema_model']) instead
of waiting for a new "best" snapshot, and renders a handful of real peri
photos (seoul_univ_inspect_COCO val, filename contains "peri").

No baseline column: the deployed-baseline checkpoint path referenced by
eval_peri_fusion_progress.py / eval_peri_subset.py
(outputs/pretrained/segmentation/.../panoramic_overlapfix/checkpoint_best_total.pth)
no longer exists on this machine.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_peri_fusion_live.py
"""
import copy
from pathlib import Path

import torch

from rfdetr.scripts.compare_core import ROOT, load_coco, run_inference
from rfdetr.scripts.eval_peri_fusion_progress import render

LIVE_CKPT = ROOT / 'outputs/peri_data_fusion/checkpoint.pth'
PERI_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_COCO')
OUT_DIR = ROOT / 'outputs/peri_data_fusion/sample_renders'
PERI_RENDER_IDS = [103, 132, 163, 10, 46]


def materialize_ema_checkpoint(live_ckpt_path, tmp_path):
    """compare_core.load_model() reads checkpoint['model']; the live
    checkpoint's current-best weights are under checkpoint['ema_model']
    instead, so write a small derived checkpoint with that swapped in."""
    ckpt = torch.load(live_ckpt_path, map_location='cpu', weights_only=False)
    epoch = ckpt['epoch']
    derived = dict(model=ckpt['ema_model'], args=copy.deepcopy(ckpt['args']), epoch=epoch)
    torch.save(derived, tmp_path)
    return epoch


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = OUT_DIR / '_live_ema_tmp.pth'
    epoch = materialize_ema_checkpoint(LIVE_CKPT, tmp_ckpt)
    name = f'peri_fusion_live_ep{epoch}_ema'
    print(f'checkpoint.pth is currently at epoch {epoch}; using its ema_model weights')

    coco = load_coco(PERI_DATA / 'val', PERI_DATA / 'val/annotations.json', filename_contains='peri')
    print(f'real peri subset: {len(coco.imgs)} images, {len(coco.anns)} annotations')

    detections, rows_list = run_inference(name, tmp_ckpt, coco, PERI_DATA / 'val', OUT_DIR)
    rows = {name: {r['image_id']: r for r in rows_list}}
    dets = {name: detections}

    img_dir = OUT_DIR / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for image_id in PERI_RENDER_IDS:
        if image_id not in coco.imgs:
            print(f'skip image_id {image_id}: not in filtered peri subset')
            continue
        out_path = img_dir / f'peri_{image_id}_ep{epoch}_live.jpg'
        render(coco, dets, rows, [name], image_id, out_path, tag='(real peri photo, live ckpt)')
        rendered.append(out_path)

    tmp_ckpt.unlink(missing_ok=True)
    print('\nrendered:')
    for p in rendered:
        print(' ', p)


if __name__ == '__main__':
    main()
