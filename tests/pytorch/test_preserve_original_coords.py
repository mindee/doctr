import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def predictor():
    from doctr.models import ocr_predictor

    return ocr_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=True,
    )


@pytest.mark.parametrize(
    "angle, shape",
    [
        (12, (800, 600)),
        (-12, (800, 600)),
        (12, (600, 800)),
        (-12, (600, 800)),
    ],
)
def test_preserve_original_coords_roundtrip(angle, shape, predictor):
    """End‑to‑end: render text, skew it, run OCRPredictor with
    preserve_original_coords=True, and assert that each detected word's remapped
    box overlaps with its merged per‑word ground truth at mean IoU > 0.4.
    """
    from shapely.geometry import Polygon

    h, w = shape
    words = [("Sensitive", 50, h // 2 - 2), ("information", 185, h // 2 - 2)]

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for text, x, y in words:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # ---- Build per‑word GT using getTextSize + ink tightening ----
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    gt_orig = []
    for text, x, y in words:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, x1 = max(0, x - 5), min(w, x + tw + 5)
        y0, y1 = max(0, y - th - 5), min(h, y + bl + 5)
        region = thresh[y0:y1, x0:x1]
        ys, xs = np.nonzero(region)
        gt_orig.append((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1))

    # ---- Skew and run predictor ----
    m_skew = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, m_skew, (w, h), borderValue=(255, 255, 255))
    img_rgb = skewed  # already RGB

    # Transform GT boxes into the skewed coordinate frame so they match
    # what preserve_original_coords returns (original = the image we fed the predictor).
    # Each axis-aligned GT box becomes a rotated polygon after m_skew.
    gt_skewed = []
    for gx1, gy1, gx2, gy2 in gt_orig:
        corners = np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        homo = np.column_stack([corners, ones])
        xs, ys = (homo @ m_skew.T)[:, 0], (homo @ m_skew.T)[:, 1]
        gt_skewed.append(np.column_stack([xs, ys]))

    result = predictor([img_rgb])

    # page.page must hold the ORIGINAL (skewed) input, not the straightened image
    assert result.pages[0].page.shape[:2] == (h, w), "page.page should have original dimensions"
    assert np.array_equal(result.pages[0].page, img_rgb), "page.page should be the original input image"

    # ---- Collect detection polygons (in original coords, i.e. skewed frame) ----
    det_polys = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                g = np.array(word.geometry).reshape(-1, 2)
                det_polys.append(np.array([[g[i, 0] * w, g[i, 1] * h] for i in range(4)], dtype=np.float32))

    # ---- IoU: each detection vs its nearest ground-truth box ----
    ious = []
    for det in det_polys:
        p_det = Polygon(det)
        best_iou = 0.0
        for polygon in gt_skewed:
            p_gt = Polygon(polygon)
            if not p_gt.is_valid or p_gt.area == 0:
                continue
            union = p_det.union(p_gt).area
            if union > 0:
                best_iou = max(best_iou, p_det.intersection(p_gt).area / union)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious))
    assert mean_iou > 0.4, f"Mean IoU {mean_iou:.3f} below 0.4 threshold"


def test_preserve_original_coords_2point():
    """assume_straight_pages=True stores boxes as 2-point geometry.
    The remap loop must expand to 4 corners before the affine
    transform and return the axis-aligned envelope.
    """
    from shapely.geometry import Polygon

    from doctr.models import ocr_predictor

    h, w = 800, 600
    words = [("Sensitive", 50, 298), ("information", 185, 298)]

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for text, x, y in words:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    gt_orig = []
    for text, x, y in words:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, x1 = max(0, x - 5), min(w, x + tw + 5)
        y0, y1 = max(0, y - th - 5), min(h, y + bl + 5)
        region = thresh[y0:y1, x0:x1]
        ys, xs = np.nonzero(region)
        gt_orig.append((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1))

    angle = 12
    m_skew = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, m_skew, (w, h), borderValue=(255, 255, 255))
    img_rgb = skewed

    gt_skewed = []
    for gx1, gy1, gx2, gy2 in gt_orig:
        corners = np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        homo = np.column_stack([corners, ones])
        xs, ys = (homo @ m_skew.T)[:, 0], (homo @ m_skew.T)[:, 1]
        gt_skewed.append(np.column_stack([xs, ys]))

    predictor = ocr_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=True,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=True,
    )
    result = predictor([img_rgb])

    det_polys = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                g = np.array(word.geometry).reshape(-1, 2)
                assert g.shape[0] == 2, "2-point mode must return 2-point geometry"
                x0, y0 = g[0]
                x1, y1 = g[1]
                det_polys.append(
                    np.array([[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]], dtype=np.float32)
                )

    ious = []
    for det in det_polys:
        p_det = Polygon(det)
        best_iou = 0.0
        for polygon in gt_skewed:
            p_gt = Polygon(polygon)
            if not p_gt.is_valid or p_gt.area == 0:
                continue
            union = p_det.union(p_gt).area
            if union > 0:
                best_iou = max(best_iou, p_det.intersection(p_gt).area / union)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious))
    assert mean_iou > 0.4, f"Mean IoU {mean_iou:.3f} below 0.4 threshold"
