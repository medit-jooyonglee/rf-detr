"""Panoramic X-ray FDI 번호 후처리 (중복/outlier 보정).

docs/teeth_duplicate_post_process.md 참고.
Tuzoff et al., "Tooth detection and numbering in panoramic radiographs using
convolutional neural networks" (Sci Rep 2019) 의 heuristic 처럼, 같은 악궁(arch)의
치아는 x 좌표 순서대로 번호가 단조 증가해야 한다는 제약 아래에서 클래스 확률의
합(log)을 최대화하는 번호 배정을 찾는다.
teeth-segmentation-3d 의 PostProcessingFdinumbering(template_matching combination) 과
같은 최적해를 combination 전수 탐색 대신 DP 로 O(m * 16) 에 구한다.

label 은 universal numbering (xraypanoramic.universal_to_fdi) 기준:
    1~16  : 상악, 영상 왼쪽 -> 오른쪽 (18 ... 11, 21 ... 28)
    17~32 : 하악, 영상 오른쪽 -> 왼쪽 (38 ... 31, 41 ... 48)
"""
import numpy as np

NUM_ARCH_TEETH = 16
UPPER_LABELS = np.arange(1, 17)
# 하악은 x 증가 방향으로 label 이 감소하므로 뒤집어서 x 오름차순과 맞춘다.
LOWER_LABELS = np.arange(32, 16, -1)


def _monotonic_assignment(log_probs):
    """log_probs (m, K) 에서 position 이 strictly increasing 인 최대 합 배정을 반환."""
    m, num_positions = log_probs.shape
    dp = np.full((m, num_positions), -np.inf)
    back = np.zeros((m, num_positions), dtype=np.int64)
    dp[0] = log_probs[0]
    for i in range(1, m):
        # prefix_best[k] = max(dp[i-1][:k]) , prefix_arg 는 그 위치
        prev = dp[i - 1]
        prefix_arg = np.zeros(num_positions, dtype=np.int64)
        prefix_best = np.full(num_positions, -np.inf)
        best, best_arg = -np.inf, 0
        for k in range(num_positions):
            prefix_best[k], prefix_arg[k] = best, best_arg
            if prev[k] > best:
                best, best_arg = prev[k], k
        dp[i] = log_probs[i] + prefix_best
        back[i] = prefix_arg
    positions = np.zeros(m, dtype=np.int64)
    positions[-1] = int(np.argmax(dp[-1]))
    for i in range(m - 1, 0, -1):
        positions[i - 1] = back[i, positions[i]]
    return positions


def refine_fdi_labels(labels, probs, boxes_xyxy, eps=1e-6, verbose=True):
    """상/하악별 x 정렬 기반으로 중복·outlier label 을 보정한다.

    Args:
        labels: (N,) universal label (1~32).
        probs: (N, C) sigmoid class 확률, C >= 33 (index 0 은 background).
        boxes_xyxy: (N, 4) 박스. x 중심으로 정렬에 사용.

    Returns:
        (N,) 보정된 label. argmax 결과가 이미 단조 증가(중복 없음)이면 그대로 유지된다.
    """
    labels = np.asarray(labels)
    refined = labels.copy()
    if labels.size == 0 or probs.shape[-1] < 33:
        return refined
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5

    for arch_name, arch_labels in (('upper', UPPER_LABELS), ('lower', LOWER_LABELS)):
        members = np.where(np.isin(labels, arch_labels))[0]
        # 16개 초과면 단조 배정이 불가능하므로 (과검출) 손대지 않는다.
        if members.size < 2 or members.size > NUM_ARCH_TEETH:
            continue
        members = members[np.argsort(centers_x[members], kind='stable')]
        log_probs = np.log(np.clip(probs[members][:, arch_labels], eps, 1.0))
        positions = _monotonic_assignment(log_probs)
        new_labels = arch_labels[positions]
        if verbose and np.any(new_labels != labels[members]):
            from rfdetr.datasets.xraypanoramic import label_to_fdi
            print(f'[fdi-refine] {arch_name}: '
                  f'{label_to_fdi(labels[members]).tolist()} -> {label_to_fdi(new_labels).tolist()}')
        refined[members] = new_labels
    return refined
