RF-DETR Codebase Map & Code Tricks
All paths are relative to d:\workspace\repositories\rf-detr_main\rfdetr. Line numbers are as of the version currently checked out.

1. Top-level module map
detr.py (487 lines) — the public, user-facing API. RFDETR base class (detr.py:43) wraps a Model instance from main.py. Its __init__ (detr.py:53-65) builds a ModelConfig (or a subclass config), downloads pretrained weights, and constructs self.model = Model(**config.dict()). Subclasses RFDETRBase/Large/Nano/Small/Medium/SegPreview (detr.py:426-487) just override get_model_config/get_train_config to point at different pydantic config classes from config.py. Key methods: train() (detr.py:79-84, 127-204) assembles callbacks (metrics sinks, early stopping) and calls self.model.train(...); predict() (detr.py:232-368) does preprocessing (resize/normalize) and calls either the raw model or a torch.jit.traced "optimized" model (optimize_for_inference, detr.py:86-109); export() (detr.py:119-125) delegates to Model.export in main.py.

main.py (1098 lines) — the Model class (main.py:78) is the training driver: it calls build_model(args) (from lwdetr.py) to construct the nn.Module, handles checkpoint loading with class-count mismatch resizing (main.py:96-142), optional LoRA injection into the backbone (main.py:144-156), and holds PostProcess. Model.train() (main.py:168-530) is the actual epoch loop: builds optimizer/param groups (via get_param_dicts.py), LR schedule (lr_lambda, main.py:222-236), datasets/dataloaders, EMA (ModelEma, main.py:284-287), drop-path/dropout schedules (main.py:328-338), then per-epoch calls train_one_epoch/evaluate from engine.py. Model.export() (main.py:532-602) drives ONNX export via rfdetr/deploy/export.py. populate_args/get_args_parser (main.py:689-1099) convert kwargs into an argparse.Namespace consumed everywhere else as args.

models/lwdetr.py (879 lines) — the actual nn.Module wiring. build_model(args) (lwdetr.py:775-834) instantiates backbone (via build_backbone), transformer (via build_transformer), optional SegmentationHead, then the LWDETR module (lwdetr.py:40) which owns class_embed/bbox_embed heads, refpoint_embed/query_feat embeddings (group-DETR-sized), and wires self.transformer.decoder.bbox_embed = self.bbox_embed for iterative box refinement (lwdetr.py:82-86). LWDETR.forward (lwdetr.py:132-215) runs backbone → transformer → heads → aux/enc outputs. build_criterion_and_postprocessors(args) (lwdetr.py:836-879) builds HungarianMatcher + SetCriterion (lwdetr.py:282) + PostProcess (lwdetr.py:709).

models/transformer.py (591 lines) — Transformer (transformer.py:129) implements the two-stage encoder-output-proposal logic (gen_encoder_output_proposals, transformer.py:72-126) and calls TransformerDecoder (transformer.py:306), whose layers are TransformerDecoderLayer (transformer.py:445) combining plain nn.MultiheadAttention self-attn with MSDeformAttn cross-attn (imported from models/ops/modules). Note: there is no encoder module here — self.encoder = None is hardcoded (transformer.py:141); the ViT backbone itself serves as the "encoder."

models/backbone/: backbone.py — Backbone class (backbone.py:36) parses encoder name string (e.g. dinov2_windowed_small) into registers/windowed flags (backbone.py:64-75), builds a DinoV2 encoder + MultiScaleProjector (from projector.py); __init__.py — Joiner (__init__.py:20) is an nn.Sequential(backbone, position_embedding) that also produces positional embeddings per feature level, and build_backbone (__init__.py:54) assembles it. dinov2.py — DinoV2 (dinov2.py:48) chooses between plain HF AutoBackbone (no windowing) and the custom WindowedDinov2WithRegistersBackbone (windowed attention + register tokens), and defines the export() position-embedding-interpolation trick (dinov2.py:125-183). dinov2_with_windowed_attn.py — full from-scratch reimplementation of HF's Dinov2-with-registers with window partitioning built into embeddings/encoder (dinov2_with_windowed_attn.py:216-329, 652-707).

models/matcher.py — HungarianMatcher (matcher.py:32), Hungarian assignment combining focal-style class cost, L1 box cost, GIoU cost, and (if segmentation) mask BCE/Dice cost, solved per "group" via scipy.optimize.linear_sum_assignment.

models/segmentation_head.py — SegmentationHead (segmentation_head.py:62) does query×spatial-feature dot-product mask prediction (einsum('bchw,bnc->bnhw', ...)) plus PointRend-style uncertainty point sampling (get_uncertain_point_coords_with_randomness, segmentation_head.py:149-202) used by both the matcher and the criterion.

engine.py — train_one_epoch (engine.py:50-183) and evaluate (engine.py:257-351), the actual step loops called by Model.train() in main.py.

config.py — Pydantic dataclasses: ModelConfig (config.py:13) base architecture knobs, subclassed per model size (RFDETRBaseConfig config.py:42, RFDETRLargeConfig config.py:62, RFDETRNanoConfig config.py:74, RFDETRSmallConfig config.py:86, RFDETRMediumConfig config.py:98, RFDETRSegPreviewConfig config.py:110); TrainConfig (config.py:123) and SegmentationTrainConfig (config.py:164) for training hyperparameters.

cli/main.py — this is not a generic CLI; it's a personal/experimental script (train_from_teeth_dir, train_from_xray_teeth_dir, hardcoded Windows dataset paths like E:/dataset/... at cli/main.py:106, and a __main__ block that unconditionally calls train_from_xray_teeth_dir() at cli/main.py:223-233). Treat it as a scratch/example file, not part of the stable public surface.

Wiring chain: detr.py (RFDETR) → main.py (Model) → models/lwdetr.py::build_model (backbone+transformer+heads) and build_criterion_and_postprocessors (matcher+criterion) → engine.py (train_one_epoch/evaluate) called from inside Model.train().

2. Backbone
ViT variant: DINOv2 (with optional register tokens) reimplemented with windowed attention, chosen via name string in Backbone.__init__ (models/backbone/backbone.py:64-87). Windowing is a ViTDet-style trick: most blocks run attention only within local windows, and specific blocks (out_feature_indexes) run full/global attention over the whole image.
Window/global interleaving: WindowedDinov2WithRegistersConfig.window_block_indexes (dinov2_with_windowed_attn.py:179) defaults to all blocks being windowed; dinov2.py:80-82 computes it as the complement of out_feature_indexes — i.e. blocks that produce an output feature map get full attention, all others get windowed attention. The actual toggle happens per-layer in WindowedDinov2WithRegistersEncoder.forward via run_full_attention = i not in self.config.window_block_indexes (dinov2_with_windowed_attn.py:678), passed into WindowedDinov2WithRegistersLayer.forward (dinov2_with_windowed_attn.py:603-649) which reshapes tokens to merge/split windows around the attention call (dinov2_with_windowed_attn.py:613-617, 626-631).
Windowing mechanics: window partitioning is folded straight into the patch embeddings, not a separate op — WindowedDinov2WithRegistersEmbeddings.forward reshapes/permutes the patch grid into (batch*num_windows², tokens_per_window, C) (dinov2_with_windowed_attn.py:305-319), and WindowedDinov2WithRegistersBackbone.forward undoes the windowing when producing the final feature maps (dinov2_with_windowed_attn.py:1096-1107).
Multi-scale features from a plain ViT: a single-scale ViT trunk feeds MultiScaleProjector (models/backbone/projector.py:141) which builds P3–P6-like pyramids from one feature map per out_feature_indexes entry, using ConvTranspose2d upsampling for scale 2.0/4.0 and strided conv for 0.5, then fuses via C2f blocks (YOLO-style CSP block, projector.py:123-138) — this is explicitly commented as a ViTDet derivative (projector.py:9, "Modified from ViTDet").
Positional embedding interpolation: two places implement bicubic interpolation of the DINOv2 positional embeddings for non-native resolutions: WindowedDinov2WithRegistersEmbeddings.interpolate_pos_encoding (dinov2_with_windowed_attn.py:234-286, used at both training time and export/tracing time with antialias=True bicubic resize), and a second, export-only variant baked in at DinoV2.export() (dinov2.py:125-183) which precomputes a fixed-size interpolated embedding once and monkey-patches interpolate_pos_encoding via types.MethodType (dinov2.py:171-183) so ONNX/TorchScript tracing doesn't need dynamic interpolation logic.
Patch size / config mismatch handling: DinoV2.__init__ (dinov2.py:89-99) detects when positional_encoding_size * patch_size doesn't match the DINOv2 pretrained image_size/patch_size, and in that case silently drops load_dinov2_weights and prints a warning rather than erroring — a convenience trick for custom resolutions/patch sizes.
3. Transformer / decoder tricks
Attention type: hybrid — decoder self-attention is plain nn.MultiheadAttention (transformer.py:453), decoder cross-attention is deformable (MSDeformAttn, transformer.py:458-459, implemented in models/ops/modules/ms_deform_attn.py:40). There is no transformer encoder (self.encoder = None, transformer.py:141) — the ViT backbone plays that role, feeding multi-scale memory directly to the decoder.
Query initialization: two coexisting mechanisms, selected by two_stage:
Non-two-stage: LWDETR holds learned refpoint_embed/query_feat embeddings sized num_queries * group_detr (lwdetr.py:72-75); only num_queries are used at inference (lwdetr.py:163-165).
Two-stage (RF-DETR's default, ModelConfig.two_stage=True at config.py:17): gen_encoder_output_proposals (transformer.py:72-126) turns encoder memory into per-pixel box proposals, Transformer.forward (transformer.py:225-265) scores/selects top-k per "group" via enc_out_class_embed/enc_out_bbox_embed (lwdetr.py:102-105), and merges the selected content ("memory_ts") and positions ("boxes_ts") with the plain learned query embeddings (transformer.py:267-286) — a Deformable-DETR-style two-stage query init, group-DETR-aware.
Iterative box refinement: implemented in TransformerDecoder.forward (transformer.py:343-442). Each layer (unless lite_refpoint_refine) recomputes reference points via bbox_embed and refpoints_refine (transformer.py:332-341, 404-410), detaching between layers (refpoints_unsigmoid = new_refpoints_unsigmoid.detach(), transformer.py:410) — classic DAB/Deformable-DETR iterative anchor refinement. lite_refpoint_refine mode (transformer.py:374-380) skips per-layer refinement entirely, computing reference embeddings once, for speed.
bbox_reparam trick: instead of predicting (box + ref).sigmoid(), boxes are reparameterized as offset+scale relative to the reference box in a delta/exp form (lwdetr.py:171-177, mirrored in transformer.py:235-241 and refpoints_refine at transformer.py:333-338) — i.e. cxcy = delta_cxcy * ref_wh + ref_cxcy, wh = exp(delta_wh) * ref_wh, avoiding sigmoid saturation issues. Both RFDETRBaseConfig default this to True (config.py:25).
Auxiliary losses per layer: out['aux_outputs'] = self._set_aux_loss(...) (lwdetr.py:189-190), consumed in SetCriterion.forward (lwdetr.py:562-572), one Hungarian match + loss set per decoder layer.
Denoising (DN) queries: absent. A repo-wide search for "dn"/"denoising" found nothing — this codebase does not implement DN-DETR/DINO-style denoising training, unlike many contemporary DETR variants.
Group DETR: a distinctive trick throughout — group_detr (default 13, main.py:761) replicates queries into independent groups during training only (self-attention is batched per-group via q.split(...)/torch.cat(..., dim=0) in TransformerDecoderLayer.forward_post, transformer.py:499-511), each group gets its own Hungarian match (matcher.py:151-165 splits the cost matrix by g_num_queries), speeding convergence; at inference only the first num_queries slice is used (lwdetr.py:163-165, 220-221).
4. Matcher (models/matcher.py)
HungarianMatcher.forward (matcher.py:57-165) combines three (or five, with masks) costs into one matrix C:
Classification cost: a focal-loss-style pos/neg cost computed with F.logsigmoid for numerical stability (matcher.py:110-112), not raw log/pow — comment at matcher.py:109 notes the refactor is deliberately for stability.
L1 box cost: torch.cdist(out_bbox, tgt_bbox, p=1) (matcher.py:115).
GIoU cost: -generalized_box_iou(...) (matcher.py:95-96).
If segmentation head active: mask BCE cost (batch_sigmoid_ce_loss) and Dice cost (batch_dice_loss), both computed on randomly sampled points rather than the full mask grid (matcher.py:127-139) — a PointRend-style efficiency trick to avoid O(H·W) cost per pair.
Efficiency/robustness tricks:
Early-exits with empty assignment lists if there are zero total target boxes across the batch (matcher.py:102-106), avoiding a crash in linear_sum_assignment.
C = C.view(bs, num_queries, -1).float().cpu() (matcher.py:145) — cost matrix explicitly cast to float32 and CPU before Hungarian solve because "bfloat16 doesn't play nicely with CPU" (comment at matcher.py:145) and scipy needs numpy/CPU tensors.
NaN/Inf costs are replaced with 2 * max_cost rather than left as-is (matcher.py:148-149), preventing linear_sum_assignment failures on degenerate boxes.
Group-DETR batching: cost matrix is split along the query dimension into group_detr chunks (matcher.py:153-154), each solved with an independent linear_sum_assignment per group per image (matcher.py:155-164), then indices re-offset and concatenated — this is a per-image, per-group loop (not vectorized across the batch), which is the standard (if not maximally efficient) way scipy's assignment is used in DETR-family repos.
5. Loss / Criterion (models/lwdetr.py::SetCriterion)
Losses computed (SetCriterion.get_loss dispatch table, lwdetr.py:524-532): labels (classification), boxes (L1 + GIoU), cardinality (logging-only, no gradient — decorated @torch.no_grad() at lwdetr.py:419), and conditionally masks (BCE + Dice via point sampling, lwdetr.py:454-509, mirroring the PointRend recipe used in the matcher: get_uncertain_point_coords_with_randomness + point_sample from segmentation_head.py).
Classification loss has four selectable variants inside loss_labels (lwdetr.py:322-417), toggled by config flags:
Standard sigmoid focal loss (sigmoid_focal_loss, lwdetr.py:589-614), alpha=focal_alpha (default 0.25, config.py/main.py:785,945), gamma=2.
use_varifocal_loss → sigmoid_varifocal_loss (lwdetr.py:617-625), targets are IoU-weighted (lwdetr.py:385-400).
use_position_supervised_loss → position_supervised_loss (lwdetr.py:628-637), IoU-weighted targets normalized by per-query max (lwdetr.py:364-383).
ia_bce_loss (IoU-aware BCE, RF-DETR's default via ModelConfig.ia_bce_loss=True at config.py:36) → a fused, numerically-stable reformulation using F.logsigmoid directly rather than computing prob.log()/(1-prob).log() separately (lwdetr.py:359-362), with a try/except: pass around the positive-weight computation (lwdetr.py:352-358) — a defensive-but-silent trick to avoid crashing on edge cases (e.g. index errors from empty matches).
num_boxes normalization (SetCriterion.forward, lwdetr.py:547-554): sums target box counts across the batch, multiplies by group_detr unless sum_group_losses is set, wraps in a tensor, does torch.distributed.all_reduce if distributed is initialized, divides by get_world_size(), and clamps to min=1 — the standard DETR distributed-normalization trick so loss scales are consistent regardless of GPU count/batch distribution.
Prior-probability bias init for focal loss: LWDETR.__init__ sets class_embed.bias to -log((1-0.01)/0.01) (lwdetr.py:90-93) so that at initialization the sigmoid output is ~0.01 for all classes — standard RetinaNet-style focal-loss bias trick to prevent loss explosion early in training.
reinitialize_detection_head (lwdetr.py:109-122): resizes the classification head to a new num_classes by repeating the existing weight/bias tensor (repeat(num_repeats, 1) then truncating), rather than reinitializing from scratch — used both for continuing training with a different class count and for temporarily matching a checkpoint's head size in main.py:104-142 before resizing back down.
6. Distinctive engineering patterns ("code tricks")
EMA model: ModelEma (util/utils.py:8-44) is a deepcopyd shadow model updated via _update with a lambda decay*e + (1-decay)*m (util/utils.py:30-41); optionally uses a time-constant-based decay ramp-up (_get_decay, util/utils.py:23-28: decay * (1 - exp(-updates/tau))) instead of a fixed decay from step 1, so early updates aren't dominated by random init. Called every step of train_one_epoch (engine.py:172-174) after the optimizer step.
NestedTensor pattern: NestedTensor (util/misc.py:312-333) bundles a batched, zero-padded image tensor with a boolean padding mask; nested_tensor_from_tensor_list (util/misc.py:335-357) computes _max_by_axis over the batch and pads/masks each image (util/misc.py:344-354), with a separate ONNX-traceable code path (_onnx_nested_tensor_from_tensor_list, util/misc.py:362+) since the padding loop doesn't trace. The mask travels alongside features into position encodings, deformable attention (memory_key_padding_mask), and mask interpolation everywhere in Backbone.forward (models/backbone/backbone.py:121-135).
_set_aux_loss TorchScript workaround: LWDETR._set_aux_loss (lwdetr.py:253-263) is decorated @torch.jit.unused with an explicit comment explaining it exists because TorchScript can't handle a dict with heterogeneous value types (tensor vs list) — a well-known DETR-family idiom, replicated here verbatim.
Drop-path / dropout schedules updated per-step: drop_scheduler (util/drop_scheduler.py:10-33) precomputes a per-iteration array (constant/linear, early/late cutoff modes); train_one_epoch looks up the current iteration's value and pushes it into the live model via model.update_drop_path(...)/model.update_dropout(...) (engine.py:100-111), which in turn walk self.backbone[0].encoder.blocks[i].drop_path.drop_prob (lwdetr.py:265-274) or every nn.Dropout in the transformer (lwdetr.py:276-279) and mutate .drop_prob/.p in place mid-training — a schedule implemented via live attribute mutation rather than a wrapped module.
Multi-scale training via on-the-fly interpolation: in train_one_epoch (engine.py:113-125), if args.multi_scale is set, a random scale is chosen deterministically per-step (random.seed(it) then random.choice(scales), engine.py:115-116 — seeding by step index rather than using a persistent RNG, so it's reproducible across restarts/ranks) and the entire batch (images, masks, and segmentation masks) is resized with F.interpolate inside torch.inference_mode() (engine.py:117-125) before being split into gradient-accumulation sub-batches. Candidate scales come from compute_multi_scale_scales (datasets/coco.py:31-38), which rounds to multiples of patch_size * num_windows so the resized image stays compatible with ViT patching and window attention.
export()/forward_export duplication pattern: pervasive throughout the codebase — nearly every module (LWDETR.export/forward_export at lwdetr.py:124-251, Backbone.export/forward_export at models/backbone/backbone.py:112-149, Joiner.export/forward_export at models/backbone/__init__.py:33-51, SegmentationHead.export/forward_export at segmentation_head.py:78-120, PositionEmbeddingSine.export/forward_export at position_encoding.py:43-98, TransformerDecoder/MSDeformAttn/DinoV2 all have _export flags) implements a self-modifying export switch: export() sets self._export = True, stashes the original forward as self._forward_origin, and monkey-patches self.forward = self.forward_export (e.g. lwdetr.py:124-130), then recursively calls .export() on any submodule exposing that interface via hasattr(m, "export") reflection (same pattern repeated at lwdetr.py:128-130, models/backbone/__init__.py:37-44, segmentation_head.py:82-84). This avoids if self._export: ... branches littering every forward pass and keeps the ONNX/TorchScript-traced path free of the NestedTensor/masking/dict-output logic that doesn't trace well. RFDETR.optimize_for_inference (detr.py:86-109) is the caller that triggers this cascade before torch.jit.trace.
Gradient accumulation loop structure: in train_one_epoch (engine.py:126-145), a single "step" batch is sliced into args.grad_accum_steps sub-batches, each run in its own autocast context with loss scaled by 1/grad_accum_steps and immediately .backward()'d (scaler.scale(losses).backward(), engine.py:145) inside the loop, but the optimizer step/zero_grad/EMA update only happen once after all sub-batches (engine.py:164-174) — standard grad-accum, notable mainly because the multi-scale resize (engine.py:113-125) happens once per outer step, not per sub-batch, so all accumulated micro-batches within one step share the same random resolution.
Mixed precision: autocast/GradScaler compatibility shim for old/new torch APIs (get_autocast_args, engine.py:33-47) — tries torch.amp (new) first, falls back to torch.cuda.amp with a DEPRECATED_AMP flag; always uses bfloat16 (engine.py:45,47), never fp16, for autocast, but supports a separate manual fp16 path for eval only via args.fp16_eval (model/samples explicitly .half()'d and outputs cast back to float per-key including nested aux_outputs/enc_outputs, engine.py:278-300).
Callbacks system: a defaultdict(list) of named hooks (RFDETR.callbacks, detr.py:57) restricted to a fixed whitelist (currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"], main.py:169) that raises ValueError on unknown keys (main.py:170-175) — on_train_batch_start fires once per outer step with {step, model, epoch} (engine.py:93-99), on_fit_epoch_end fires with the full log_stats dict (main.py:475-476), on_train_end fires with no args (main.py:529-530). Metrics sinks (plot/TensorBoard/W&B) and early stopping are wired onto these hooks in RFDETR.train_from_config (detr.py:171-199) rather than being hardcoded into the training loop.
LR-decay-by-name pattern: rather than a config-driven per-layer LR list, both get_dinov2_lr_decay_rate (models/backbone/backbone.py:176-193) and the near-duplicate get_vit_lr_decay_rate (util/get_param_dicts.py:13-31) parse the parameter's string name (looking for substrings like ".layer." or ".blocks.") to infer which transformer depth it belongs to and apply lr_decay_rate ** (num_layers+1-layer_id) — a fragile-but-common "layer-wise LR decay via name matching" trick; weight-decay-by-name similarly zeroes WD for any param whose name contains gamma/pos_embed/rel_pos/bias/norm/embeddings (models/backbone/backbone.py:195-205).
7. How to approach reading this codebase
Recommended reading order (matches the actual call chain, narrow → deep):

detr.py — read top-to-bottom, it's short and mostly glue. Understand RFDETR.__init__, .train(), .predict(). Skim the size-subclasses at the bottom (detr.py:426-487); they're nearly identical boilerplate.
config.py — read all of it before touching main.py; every "gotcha" default (e.g. two_stage=True, bbox_reparam=True, ia_bce_loss=True, group_detr=13) that changes which code branches execute lives here, not in main.py's populate_args defaults (which mostly exist for the standalone-script path and get overridden).
main.py — skim populate_args/get_args_parser (they're 400+ lines of nearly mechanical argument plumbing, not worth reading line-by-line — just grep for a flag when you need its default). Read Model.__init__ and Model.train carefully instead; this is the real orchestration logic (dataset/sampler construction, EMA, checkpoint save/resume logic, LR schedule).
models/lwdetr.py — read build_model, then LWDETR.forward (lwdetr.py:132-215), then SetCriterion (loss math). This file is the hub; almost every other module is imported here.
models/transformer.py — the hardest file. Read gen_encoder_output_proposals and Transformer.forward's two-stage branch first (this is where query init happens), then TransformerDecoder.forward's per-layer refinement loop. The self-attention "group splitting" (transformer.py:499-511) is confusing in isolation — read it only after you understand group_detr from lwdetr.py.
models/backbone/ — read backbone.py (Backbone class) and __init__.py (Joiner/build_backbone) first for the wiring, then dinov2.py for how the encoder is selected/configured, and only dive into dinov2_with_windowed_attn.py (1129 lines, mostly copied HF boilerplate with windowing spliced in) if you specifically need to understand windowed attention internals — search for run_full_attention and num_windows rather than reading linearly.
models/matcher.py and models/segmentation_head.py — short, self-contained, safe to read fully; they're the least entangled with the rest of the codebase.
engine.py — read after main.py::Model.train, since it's called from there; the multi-scale/grad-accum/AMP logic in train_one_epoch is dense but linear.
Safe to skim / not part of the core architecture:

cli/main.py is a leftover personal/experimental script (hardcoded local paths, an unconditional train_from_xray_teeth_dir() call at the bottom) — don't treat it as documentation of intended CLI usage.
main.py's get_args_parser/populate_args (main.py:689-1099) and the if __name__ == '__main__': block (main.py:605-687) are a legacy standalone-script entry point; the pydantic configs in config.py are the actual modern interface used by detr.py.
The extensive HF docstrings/examples inside dinov2_with_windowed_attn.py (e.g. dinov2_with_windowed_attn.py:56-121, 753-901) are copy-pasted from upstream transformers and can be skipped.
Naming/structural gotchas:

Every exportable module has a duplicate forward/forward_export pair plus an export() method that monkey-patches self.forward. If you're debugging inference behavior, always check which forward is currently bound (self._export flag) — don't assume forward is what runs after .optimize_for_inference()/.export() has been called.
num_classes off-by-one: everywhere in lwdetr.py/matcher.py, num_classes passed to build_model/SetCriterion is args.num_classes + 1 (lwdetr.py:784, 862, 870) — the "+1" background/no-object convention from original DETR; don't be surprised by mismatched class counts when cross-referencing ModelConfig.num_classes.
Two nearly-identical LR-decay-by-parameter-name functions exist (models/backbone/backbone.py:176-205 for DINOv2, util/get_param_dicts.py:13-38 for a generic/legacy ViT) — only the former is actually used by the current DINOv2 backbone path; the latter looks vestigial from a prior (non-HF) ViT implementation.
Config class inheritance chain matters for defaults: RFDETRLargeConfig/Nano/Small/Medium/SegPreview (config.py:62-121) all subclass RFDETRBaseConfig, not ModelConfig directly, so a change to a Base default silently propagates to all size variants unless overridden — check the MRO when a size variant behaves unexpectedly.
args is a loosely-typed argparse.Namespace, not the pydantic config: populate_args (main.py:877-1099) re-derives an args namespace from kwargs inside Model.__init__/Model.train, separate from RFDETR.model_config/get_train_config's pydantic objects in detr.py. Extra/unknown kwargs are silently absorbed via **extra_kwargs (main.py:999) — a typo'd kwarg won't raise, it'll just vanish into args unused, which is one of the easier ways to silently misconfigure a training run.
No DN (denoising) queries despite this being a common feature in comparable DETR variants (DINO, DN-DETR) — don't go looking for it; the query-quality mechanism here is purely two-stage proposal selection + group-DETR + iterative refinement.
탐색이 끝났습니다. 실제 코드를 다 확인한 내용을 바탕으로 정리하겠습니다.

1. 모듈 구성 (호출 체인)

detr.py (RFDETR, 사용자 API)
  └─ main.py (Model, 학습 드라이버)
       ├─ models/lwdetr.py :: build_model()          → backbone + transformer + heads 조립
       ├─ models/lwdetr.py :: build_criterion_and_postprocessors() → matcher + criterion
       └─ engine.py :: train_one_epoch() / evaluate() → 실제 step 루프
detr.py: 퍼블릭 API. RFDETR.__init__이 config.py의 pydantic config로 Model을 생성. train()/predict()/export()는 대부분 main.py/deploy/export.py로 위임. 아래쪽 RFDETRBase/Large/Nano/Small/Medium/SegPreview 클래스들(detr.py:426-487)은 거의 동일한 보일러플레이트 — config 클래스만 바뀜.
main.py: Model 클래스가 실제 학습 드라이버. 체크포인트 로드 시 클래스 개수 불일치 처리(main.py:96-142), LoRA 주입, optimizer/LR schedule, EMA, drop-path 스케줄 구성 후 engine.py의 루프를 호출. get_args_parser/populate_args(main.py:689-1099)는 레거시 standalone 스크립트용 — 실제 안 쓰는 부분이 많아 라인별로 읽을 필요 없음.
models/lwdetr.py: 허브. LWDETR nn.Module(backbone+transformer+heads 조립), SetCriterion(loss), PostProcess가 다 여기 있음.
models/transformer.py: 가장 복잡한 파일. two-stage proposal, iterative box refinement, group-DETR self-attention 분기가 다 여기.
models/backbone/: backbone.py(래퍼)+__init__.py(Joiner)+dinov2.py/dinov2_with_windowed_attn.py(윈도우 어텐션 DINOv2 재구현)+projector.py(ViTDet식 멀티스케일).
models/matcher.py, models/segmentation_head.py: 짧고 독립적, 전체 읽기 부담 없음.
engine.py: main.py에서 매 epoch 호출되는 실제 step 루프.
config.py: pydantic dataclass들. 여기 기본값이 진짜 동작을 결정 (two_stage=True, bbox_reparam=True, ia_bce_loss=True, group_detr=13 등).
cli/main.py: 정식 공개 인터페이스가 아니라 사용자님(?)이 실험용으로 만든 개인 스크립트에 가까움 — 하드코딩된 로컬 경로, __main__에서 무조건 train_from_xray_teeth_dir() 호출. 문서로 취급하지 마세요.
2. 핵심 구현 기교
Backbone

DINOv2를 window attention(ViTDet 스타일)으로 재구현. out_feature_indexes에 해당하는 블록만 전역(global) attention, 나머지는 로컬 윈도우 attention — 토글은 dinov2_with_windowed_attn.py:678 run_full_attention = i not in window_block_indexes.
윈도우 분할이 별도 연산이 아니라 patch embedding 단계에서 바로 reshape/permute로 접혀 들어감 (dinov2_with_windowed_attn.py:305-319).
단일 스케일 ViT에서 멀티스케일 피처를 만드는 MultiScaleProjector(projector.py:141) — ConvTranspose로 업샘플 + strided conv로 다운샘플 + YOLO식 C2f 블록으로 fuse. 코드에 "Modified from ViTDet"라고 명시.
위치 임베딩 bicubic interpolation이 학습용/export용으로 따로 두 벌 존재 — export용은 types.MethodType으로 forward를 몽키패치해서 트레이싱 시 동적 분기를 없앰 (dinov2.py:171-183).
Transformer/디코더

인코더가 없음 — self.encoder = None (transformer.py:141). ViT 백본이 인코더 역할을 대신함. 디코더 self-attn은 일반 nn.MultiheadAttention, cross-attn은 deformable attention.
Two-stage query 초기화: 인코더 피처에서 픽셀별 박스 proposal을 만들고 top-k를 선택해 learnable query와 합침(Deformable-DETR 스타일) — gen_encoder_output_proposals (transformer.py:72-126).
bbox_reparam: 박스를 sigmoid로 직접 예측하지 않고 cxcy = delta*ref_wh + ref_cxcy, wh = exp(delta_wh)*ref_wh 형태로 reference box 대비 델타로 예측 — sigmoid saturation 회피 (lwdetr.py:171-177).
Group-DETR: 학습 시에만 쿼리를 group_detr(기본 13)개 그룹으로 복제, 그룹별 독립 self-attention + 독립 헝가리안 매칭으로 수렴 가속. 추론 시엔 첫 그룹만 사용 (transformer.py:499-511, matcher.py:151-165).
DN(디노이징 쿼리)은 아예 없음 — DINO/DN-DETR과 달리 이 구현엔 없는 기능. 찾으려 하지 마세요.
Matcher

클래스 비용을 F.logsigmoid로 수치안정하게 계산 (matcher.py:110-112).
마스크 비용은 전체 grid가 아니라 랜덤 샘플 포인트로만 계산(PointRend식, matcher.py:127-139) — 이게 앞서 얘기한 세그멘테이션 비효율 문제의 완화 장치이기도 함.
비용 행렬을 float().cpu()로 강제 변환(주석: "bfloat16 doesn't play nicely with CPU") 후 scipy linear_sum_assignment 호출, NaN/Inf는 2*max_cost로 치환해 assignment 실패 방지 (matcher.py:145-149).
Loss

분류 loss가 4가지 변형 중 config로 선택됨: 표준 focal / varifocal / position-supervised / ia_bce(기본값) — 전부 lwdetr.py:322-417.
focal loss 초기 bias를 -log(0.99/0.01)로 세팅해서 초기 sigmoid 출력이 ~0.01 되게 함(RetinaNet 트릭, lwdetr.py:90-93).
num_boxes 정규화 시 all_reduce로 분산 학습 GPU 개수와 무관하게 loss 스케일 유지 (lwdetr.py:547-554).
reinitialize_detection_head: 클래스 수를 바꿀 때 head 가중치를 초기화가 아니라 반복(repeat) 후 자르기로 리사이즈 (lwdetr.py:109-122).
그 외 눈에 띄는 엔지니어링 패턴

export()/forward_export 몽키패치 패턴: 거의 모든 서브모듈이 self.forward = self.forward_export로 바꿔치기하는 export() 메서드를 갖고 있고, 이게 재귀적으로(hasattr(m, "export")) 전파됨. if self._export: ... 분기를 forward 안에 안 두려는 목적 — ONNX/TorchScript 트레이싱 경로를 깔끔하게 분리.
NestedTensor: 배치 내 이미지 크기가 다를 때 zero-padding + bool mask로 묶는 패턴. 방금 고친 세그멘테이션 버그가 바로 이 패딩이 마스크엔 적용 안 됐던 문제였죠.
drop_path/dropout을 라이브 속성으로 매 스텝 mutate: 래핑된 모듈이 아니라 .drop_prob/.p를 직접 덮어씀 (engine.py:100-111, lwdetr.py:265-279).
콜백 시스템: 화이트리스트(on_fit_epoch_end/on_train_batch_start/on_train_end)만 허용, 나머지는 ValueError (main.py:169-175).
레이어별 LR decay를 파라미터 이름 문자열 매칭으로 구현(.layer./.blocks. 찾아서 depth 추론) — 다소 fragile하지만 DETR 계열 공통 관행 (backbone.py:176-193).
3. 이 코드베이스를 분석하는 팁
읽는 순서 (실제 호출 체인 순서):

detr.py — 전체 훑기 (짧음)
config.py 전체를 먼저 읽으세요 — main.py보다 먼저. 여기 기본값이 어떤 코드 분기가 실행될지 결정합니다.
main.py — populate_args/get_args_parser(400줄+)는 grep으로 필요할 때만 찾아보고, Model.__init__/Model.train만 정독.
models/lwdetr.py — build_model → LWDETR.forward → SetCriterion 순.
models/transformer.py — 가장 어려운 파일. gen_encoder_output_proposals/two-stage 분기 → TransformerDecoder.forward의 per-layer refinement 순으로. self-attention의 group 분할 부분은 lwdetr.py에서 group_detr 개념을 먼저 이해한 후 보세요.
models/backbone/ — backbone.py+__init__.py(배선) → dinov2.py(설정 선택) → dinov2_with_windowed_attn.py(1129줄, 대부분 HF 보일러플레이트 복붙 + 윈도우 로직 삽입, run_full_attention/num_windows로 검색하며 필요한 부분만).
models/matcher.py, models/segmentation_head.py — 짧고 독립적, 전체 정독 가능.
engine.py — main.py::Model.train 이해 후에.
대충 훑어도 되는 곳: cli/main.py(개인 스크립트), main.py의 get_args_parser/populate_args/__main__ 블록(레거시), dinov2_with_windowed_attn.py 안의 HF 문서 docstring들.

주의할 함정들:

모든 export 가능 모듈에 forward/forward_export 쌍이 있음 — 디버깅할 때 지금 어느 forward가 바인딩돼 있는지(self._export 플래그) 꼭 확인하세요.
num_classes는 항상 +1 (배경/no-object 클래스) — lwdetr.py/matcher.py 곳곳에서 args.num_classes + 1로 쓰임.
args(argparse Namespace)와 pydantic config는 별개 객체 — populate_args가 kwargs에서 다시 args를 만듦. 오타난 kwarg는 에러 없이 그냥 무시됨(**extra_kwargs) — 조용히 잘못 설정되는 흔한 원인.
RFDETRLarge/Nano/Small/Medium/SegPreview가 전부 RFDETRBaseConfig를 상속 — Base 기본값 바꾸면 다른 사이즈 변형에도 조용히 전파됨.