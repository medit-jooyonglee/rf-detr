"""Quick qualitative check of the still-training outputs/peri_data_fusion run
on full panoramic images (own1022 val). Companion to eval_peri_fusion_live.py
-- same rationale: checkpoint_best_ema.pth is stuck at epoch 106, so this uses
the EMA weights bundled in the live checkpoint.pth instead, and skips the
baseline column since that checkpoint path no longer exists on this machine.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_pano_fusion_live.py
"""
from pathlib import Path

from rfdetr.scripts.compare_core import load_coco, run_inference
from rfdetr.scripts.eval_peri_fusion_live import OUT_DIR, materialize_ema_checkpoint, LIVE_CKPT
from rfdetr.scripts.eval_peri_fusion_progress import render

PANO_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
PANO_RENDER_IDS = [15, 20, 71, 5, 40]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = OUT_DIR / '_live_ema_tmp.pth'
    epoch = materialize_ema_checkpoint(LIVE_CKPT, tmp_ckpt)
    name = f'pano_fusion_live_ep{epoch}_ema'
    print(f'checkpoint.pth is currently at epoch {epoch}; using its ema_model weights')

    coco = load_coco(PANO_DATA / 'val', PANO_DATA / 'val/annotations.json')
    print(f'own1022 val: {len(coco.imgs)} images, {len(coco.anns)} annotations')

    detections, rows_list = run_inference(name, tmp_ckpt, coco, PANO_DATA / 'val', OUT_DIR)
    rows = {name: {r['image_id']: r for r in rows_list}}
    dets = {name: detections}

    img_dir = OUT_DIR / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for image_id in PANO_RENDER_IDS:
        if image_id not in coco.imgs:
            print(f'skip image_id {image_id}: not in own1022 val')
            continue
        out_path = img_dir / f'panorama_{image_id}_ep{epoch}_live.jpg'
        render(coco, dets, rows, [name], image_id, out_path, tag='(pano, live ckpt)')
        rendered.append(out_path)

    tmp_ckpt.unlink(missing_ok=True)
    print('\nrendered:')
    for p in rendered:
        print(' ', p)


if __name__ == '__main__':
    main()
