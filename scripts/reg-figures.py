#!/usr/bin/env python3
"""Registration result figures.

Renders the 3D overlay of each registration backend's result — the aligned
source (orange) sitting on the target (cyan) — to PNGs, then tiles the
per-algorithm views into one comparison figure per case and emits a LaTeX
snippet so the algorithm-notes PDF can show *pictures* of the alignment next to
the numeric table (docs/organize/chapters/02-experiments.tex).

Input is the per-algorithm aligned clouds written by

    scripts/reg-bench.py <source> <target> --label <case> --save-aligned DIR

(filenames `"<case>_<algo>.ply"`, e.g. `hap101-f0__crop_gicp.ply`). The target
cloud is resolved from the case name (`"<src>__<tgt>"`) by searching --data-dirs,
or pinned with --target for a single case.

Rendering is open3d's EGL OffscreenRenderer (no display needed); metrics, when a
matching `experiments/registration-results.csv` row exists (same label + algo),
are written under each panel.

Usage:
    scripts/reg-figures.py [--aligned-dir experiments/aligned]
        [--out docs/organize/figures] [--case PREFIX] [--target PATH]
        [--data-dirs experiments/data,tests/data] [--cols 3]
        [--point-size 2.5] [--width 900 --height 650] [--no-tex]

Requires numpy, open3d, matplotlib (already in the registration python env).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "experiments" / "registration-results.csv"
TEX_OUT = REPO / "docs" / "organize" / "figures" / "registration-overlays.tex"
EXPERIMENTS_CHAPTER = REPO / "docs" / "organize" / "chapters" / "02-experiments.tex"

# Colours (RGB 0..1): target stays cool/cyan, the aligned source warm/orange so
# overlap reads as "orange points hugging cyan points".
TARGET_RGB = (0.12, 0.66, 0.80)
SOURCE_RGB = (1.00, 0.50, 0.12)
BG_RGB = (1.0, 1.0, 1.0)

# A stable algorithm ordering for the grid (unknown algos sort to the end).
ALGO_ORDER = ["icp", "icp-plane", "gicp", "vgicp", "kiss", "kiss-gicp",
              "gsdf", "gsdf-gpu"]


def parse_case_algo(stem: str) -> tuple[str, str]:
    """`hap101-f0__crop_gicp` -> ("hap101-f0__crop", "gicp").

    Algos use hyphens (icp-plane, kiss-gicp, gsdf-gpu) and the case uses `__`,
    so the algorithm is always the final underscore-delimited segment.
    """
    case, _, algo = stem.rpartition("_")
    return case, algo


def resolve_target(case: str, data_dirs: list[Path], override: Path | None) -> Path | None:
    if override:
        return override if override.is_file() else None
    # case == "<src>__<tgt>"; the aligned cloud IS the transformed source, so we
    # overlay it on the target half.
    _, _, tgt = case.partition("__")
    if not tgt:
        return None
    # Try the name as-is, with a dropped direction suffix (-rev/-fwd), and with
    # hyphens swapped for underscores (hap101-f0 -> hap101_f0.ply).
    bases = [tgt]
    for suf in ("-rev", "-fwd"):
        if tgt.endswith(suf):
            bases.append(tgt[: -len(suf)])
    cands: list[str] = []
    for b in bases:
        cands += [b, b.replace("-", "_")]
    for d in data_dirs:
        for c in cands:
            p = d / f"{c}.ply"
            if p.is_file():
                return p
    return None


def load_csv_metrics() -> dict[tuple[str, str], dict]:
    """(label, algo) -> row, so a panel can show rmse/inliers/conf when known."""
    if not CSV_PATH.is_file():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r.get("label", ""), r.get("algo", ""))] = r  # last run wins
    return out


def caption_for(case: str, algo: str, metrics: dict) -> str:
    row = metrics.get((case, algo))
    if not row:
        return algo
    bits = [algo]
    if row.get("rmse"):
        bits.append(f"rmse {row['rmse']}")
    if row.get("inliers"):
        bits.append(f"in {row['inliers']}")
    if row.get("confidence"):
        bits.append(f"conf {row['confidence']}")
    conv = {"yes": "✓", "no": "✗"}.get(row.get("converged", ""), "")
    return ("[" + conv + "] " if conv else "") + "  ".join(bits)


def frame_camera(renderer, frame_geom, zoom):
    """Frame `frame_geom`'s bbox from a fixed 3/4 view (Z-up).

    Framing the TARGET (not the union) keeps the region of interest filling the
    panel even when the source is a much larger full scan being placed onto a
    small crop — that overlap is exactly where alignment quality shows.
    """
    bbox = frame_geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.asarray(bbox.get_extent())
    radius = float(np.linalg.norm(extent)) * 0.5 + 1e-6
    direction = np.array([1.0, -1.0, 0.55])
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius * zoom
    renderer.setup_camera(50.0, center, eye, [0.0, 0.0, 1.0])
    return center


def render_overlay(target_pcd, aligned_pcd, point_size, width, height, zoom):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([*BG_RGB, 1.0])
    r.scene.scene.set_sun_light([0.3, -0.5, -0.8], [1, 1, 1], 60000)

    def mat():
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(point_size)
        return m

    r.scene.add_geometry("target", target_pcd, mat())
    r.scene.add_geometry("aligned", aligned_pcd, mat())
    frame_camera(r, target_pcd, zoom)
    img = np.asarray(r.render_to_image())
    del r  # release the EGL context / filament scene before the next case
    return img


def build_grid(case, panels, out_dir, cols):
    """panels: list of (algo_caption, image ndarray). Returns the grid PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(panels)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (cap, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(cap, fontsize=8)
    legend = [Patch(color=SOURCE_RGB, label="aligned source"),
              Patch(color=TARGET_RGB, label="target")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(case.replace("__", " → "), fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = out_dir / f"{case}_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def write_tex(case_to_png: dict[str, Path]) -> None:
    lines = [
        "% AUTO-GENERATED by scripts/reg-figures.py - do not edit by hand.",
        "% One overlay figure per registration case (aligned source over target).",
    ]
    for case, png in sorted(case_to_png.items()):
        rel = png.relative_to(TEX_OUT.parent.parent).as_posix()  # relative to docs/organize
        label = case.replace("_", "\\_")
        lines += [
            r"\begin{figure}[htbp]",
            r"  \centering",
            rf"  \includegraphics[width=\linewidth]{{{rel}}}",
            rf"  \caption{{정합 결과 3D 오버레이: {label} (주황 = 정렬된 source, 청록 = target).}}",
            rf"  \label{{fig:overlay-{case}}}",
            r"\end{figure}",
            "",
        ]
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def ensure_chapter_input() -> bool:
    """Append \\input{figures/registration-overlays} to the experiments chapter."""
    if not EXPERIMENTS_CHAPTER.is_file():
        return False
    text = EXPERIMENTS_CHAPTER.read_text(encoding="utf-8")
    needle = "figures/registration-overlays"
    if needle in text:
        return False
    block = ("\n\\section{정합 결과 시각화}\n"
             "각 백엔드의 정렬 결과를 3D로 겹쳐 본 그림이다 "
             "(\\ccfile{scripts/reg-figures.py} 로 자동 생성).\n"
             "\\input{figures/registration-overlays}\n")
    EXPERIMENTS_CHAPTER.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned-dir", default=str(REPO / "experiments" / "aligned"))
    ap.add_argument("--out", default=str(REPO / "docs" / "organize" / "figures"))
    ap.add_argument("--data-dirs",
                    default=f"{REPO/'experiments'/'data'},{REPO/'tests'/'data'}")
    ap.add_argument("--case", default="", help="only render cases containing this substring")
    ap.add_argument("--target", default="", help="pin the target cloud (single-case use)")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--point-size", type=float, default=2.5)
    ap.add_argument("--zoom", type=float, default=2.2,
                    help="camera distance as a multiple of the target's radius "
                         "(smaller = tighter on the target/overlap region)")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=650)
    ap.add_argument("--no-tex", action="store_true", help="skip the LaTeX snippet")
    args = ap.parse_args()

    try:
        import open3d as o3d  # noqa: F401  (import here for a clean error message)
    except Exception as e:
        sys.exit(f"error: open3d is required ({e}); "
                 f"pip install -r backend/registration/gradient_sdf_gpu/python/requirements.txt")

    aligned_dir = Path(args.aligned_dir)
    if not aligned_dir.is_dir():
        sys.exit(f"error: no aligned dir: {aligned_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dirs = [Path(d) for d in args.data_dirs.split(",") if d]
    target_override = Path(args.target) if args.target else None
    metrics = load_csv_metrics()

    # Group aligned clouds by case.
    cases: dict[str, dict[str, Path]] = {}
    for ply in sorted(aligned_dir.glob("*.ply")):
        case, algo = parse_case_algo(ply.stem)
        if args.case and args.case not in case:
            continue
        cases.setdefault(case, {})[algo] = ply
    if not cases:
        sys.exit(f"error: no aligned *.ply found in {aligned_dir}"
                 + (f" matching '{args.case}'" if args.case else ""))

    import open3d as o3d

    case_to_png: dict[str, Path] = {}
    for case, by_algo in sorted(cases.items()):
        tgt_path = resolve_target(case, data_dirs, target_override)
        if tgt_path is None:
            print(f"skip {case}: target cloud not found "
                  f"(pass --target or add it to --data-dirs)", file=sys.stderr)
            continue
        target_pcd = o3d.io.read_point_cloud(str(tgt_path))
        if not target_pcd.has_points():
            print(f"skip {case}: empty/unreadable target {tgt_path}", file=sys.stderr)
            continue
        target_pcd.paint_uniform_color(TARGET_RGB)

        ordered = sorted(by_algo, key=lambda a: (ALGO_ORDER.index(a)
                                                 if a in ALGO_ORDER else 99, a))
        panels = []
        for algo in ordered:
            aligned_pcd = o3d.io.read_point_cloud(str(by_algo[algo]))
            if not aligned_pcd.has_points():
                print(f"  {case}/{algo}: empty aligned cloud, skipped", file=sys.stderr)
                continue
            aligned_pcd.paint_uniform_color(SOURCE_RGB)
            img = render_overlay(target_pcd, aligned_pcd, args.point_size,
                                 args.width, args.height, args.zoom)
            single = out_dir / f"{case}_{algo}.png"
            o3d.io.write_image(str(single), o3d.geometry.Image(img))
            panels.append((caption_for(case, algo, metrics), img))
            print(f"  rendered {case}/{algo}", flush=True)
        if not panels:
            continue
        grid = build_grid(case, panels, out_dir, args.cols)
        case_to_png[case] = grid
        print(f"== {case}: target {tgt_path.name}, {len(panels)} algos -> {grid.name}")

    if not case_to_png:
        sys.exit("error: nothing rendered")

    if not args.no_tex:
        write_tex(case_to_png)
        added = ensure_chapter_input()
        print(f"\nwrote {TEX_OUT.relative_to(REPO)}"
              + ("  (+ linked from 02-experiments.tex)" if added
                 else "  (chapter already links it)"))
    print(f"figures in {out_dir.relative_to(REPO)}: "
          f"{len(case_to_png)} grids + per-algo PNGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
