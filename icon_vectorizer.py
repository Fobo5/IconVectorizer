#!/usr/bin/env python3
"""
icon_vectorizer.py

Превращает растровое изображение иконки (толстая чёрная линия на светлом фоне,
или наоборот) в чистый векторный SVG:
  - находит центральную линию (skeleton) исходной обводки
  - строит граф скелета, сглаживает каждую ветвь
  - аппроксимирует сглаженную линию кривыми Безье
  - масштабирует и центрирует результат во фрейме 24x24 (иконка ~20x20)
  - экспортирует SVG с fill="none", stroke="#000000", stroke-width="1.5",
    stroke-linecap="round", stroke-linejoin="round"

Использование:
    python3 icon_vectorizer.py input.png output.svg [--invert] [--threshold 127]

Зависимости: numpy, opencv-python, scikit-image, scipy, networkx, sknw
    pip install numpy opencv-python scikit-image scipy networkx sknw
"""

import argparse
import sys
import numpy as np
import cv2
from skimage.morphology import skeletonize
import sknw


FRAME_SIZE = 24.0
ICON_SIZE = 20.0
STROKE_WIDTH = 1.5


def load_binary_mask(path: str, invert: bool, threshold: int) -> np.ndarray:
    """Load image and return a boolean mask where True = 'ink' (the icon strokes)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")

    # Handle alpha channel: treat transparent pixels as background.
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray[alpha < 10] = 255  # transparent -> treat as white background
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Denoise a touch before thresholding (kills small raster/jpeg artifacts).
    gray = cv2.medianBlur(gray, 3)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = binary < 128  # dark pixels = ink, by default

    if invert:
        mask = ~mask

    # Heuristic: the icon (foreground) should be the minority of pixels.
    # If more than half the image is "ink", the colors were probably backwards.
    if mask.mean() > 0.5:
        mask = ~mask

    # Morphological close: bridges tiny gaps from anti-aliasing/noise before skeletonizing.
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    # Remove isolated speckle noise.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    min_area = max(4, int(0.0005 * mask_u8.size))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return clean > 0


def chaikin_smooth(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin corner-cutting: turns a jagged polyline into a smooth curve-like polyline."""
    pts = points.astype(np.float64)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append(q)
            new_pts.append(r)
        new_pts.append(pts[-1])
        pts = np.array(new_pts)
    return pts


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a polyline, dropping redundant points while preserving overall shape."""
    if len(points) < 3:
        return points

    def perp_dist(pt, start, end):
        if np.allclose(start, end):
            return np.linalg.norm(pt - start)
        v1 = end - start
        v2 = start - pt
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        return np.abs(cross_z) / np.linalg.norm(v1)

    def rdp(pts):
        if len(pts) < 3:
            return pts
        start, end = pts[0], pts[-1]
        dists = [perp_dist(p, start, end) for p in pts[1:-1]]
        if not dists:
            return np.array([start, end])
        idx = int(np.argmax(dists)) + 1
        if dists[idx - 1] > epsilon:
            left = rdp(pts[: idx + 1])
            right = rdp(pts[idx:])
            return np.vstack([left[:-1], right])
        return np.array([start, end])

    return rdp(points)


def polyline_to_bezier_path(points: np.ndarray) -> str:
    """Convert a smoothed polyline into an SVG cubic-Bezier path string using
    a Catmull-Rom -> cubic Bezier conversion, so the curve passes through every
    sample point smoothly (no sharp joints)."""
    pts = points
    n = len(pts)
    if n < 2:
        return ""
    if n == 2:
        return f"M {pts[0][0]:.2f} {pts[0][1]:.2f} L {pts[1][0]:.2f} {pts[1][1]:.2f}"

    d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(n - 1):
        p0 = pts[i - 1] if i - 1 >= 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]

        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0

        d.append(
            f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}"
        )
    return " ".join(d)


def extract_paths_from_skeleton(mask: np.ndarray):
    """Skeletonize the mask, build a graph of the skeleton, and return a list of
    smoothed polylines (one per branch/edge, and one per isolated closed loop)."""
    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        return []

    graph = sknw.build_sknw(skeleton, multi=True)

    raw_polylines = []
    for (s, e, k) in graph.edges(keys=True):
        edge_pts = graph[s][e][k]["pts"]  # array of (row, col)
        if len(edge_pts) < 2:
            continue
        # (row, col) -> (x, y)
        pts_xy = edge_pts[:, ::-1].astype(np.float64)
        raw_polylines.append(pts_xy)

    # Isolated skeleton pixels not captured as graph edges (rare, tiny dots) are skipped;
    # they'd just be noise for an icon.

    smoothed_polylines = []
    for pts in raw_polylines:
        smoothed = chaikin_smooth(pts, iterations=3)
        simplified = douglas_peucker(smoothed, epsilon=0.6)
        if len(simplified) >= 2:
            smoothed_polylines.append(simplified)

    return smoothed_polylines


def normalize_and_scale(polylines, icon_size: float, frame_size: float):
    """Fit all polylines into a centered icon_size x icon_size box inside frame_size x frame_size,
    preserving aspect ratio (no separate x/y stretching)."""
    all_pts = np.vstack(polylines)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    size = max_xy - min_xy
    size[size == 0] = 1.0

    scale = icon_size / max(size[0], size[1])
    offset = (frame_size - size * scale) / 2.0

    scaled = []
    for pts in polylines:
        p = (pts - min_xy) * scale + offset
        scaled.append(p)
    return scaled


def build_svg(polylines, frame_size: float, stroke_width: float) -> str:
    path_elems = []
    for pts in polylines:
        d = polyline_to_bezier_path(pts)
        if d:
            path_elems.append(f'  <path d="{d}"/>')

    paths_str = "\n".join(path_elems)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{int(frame_size)}" height="{int(frame_size)}" viewBox="0 0 {int(frame_size)} {int(frame_size)}" fill="none" stroke="#000000" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
{paths_str}
</svg>
"""
    return svg


def vectorize(input_path: str, output_path: str, invert: bool = False, threshold: int = 127):
    mask = load_binary_mask(input_path, invert=invert, threshold=threshold)
    polylines = extract_paths_from_skeleton(mask)

    if not polylines:
        raise RuntimeError(
            "Не удалось найти линии на изображении. Проверьте контраст/порог "
            "(--threshold) или флаг --invert."
        )

    polylines = normalize_and_scale(polylines, ICON_SIZE, FRAME_SIZE)
    svg = build_svg(polylines, FRAME_SIZE, STROKE_WIDTH)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Готово: {output_path}  ({len(polylines)} путей)")


def main():
    parser = argparse.ArgumentParser(description="Растровая иконка -> чистый stroke-SVG (24x24).")
    parser.add_argument("input", help="Путь к входному изображению (png/jpg)")
    parser.add_argument("output", help="Путь к выходному SVG")
    parser.add_argument("--invert", action="store_true", help="Инвертировать маску (если фон тёмный, а линия светлая)")
    parser.add_argument("--threshold", type=int, default=127, help="Порог бинаризации 0-255 (по умолчанию 127)")
    args = parser.parse_args()

    try:
        vectorize(args.input, args.output, invert=args.invert, threshold=args.threshold)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
