---
name: figure-maker
description: Produce paper-ready figures with a single project-wide style — density-map visualizations, qualitative comparison grids, training/eval curves, and scatter plots — output as vector PDF (and PNG mirror). Use this skill whenever the user asks to plot, draw, visualize, or generate any figure for this project — ALWAYS invoke when the user says "make a figure", "plot", "visualize density map", "qualitative comparison", "training curve", "ablation chart", or anything that ends up in a paper or slide. Locks fonts, palette, DPI, and legend placement so reviewers never complain about inconsistent figure style.
---

# figure-maker

Single source of truth for how every figure in this project looks.

## When to invoke
- user asks to plot, draw, visualize, render, "make a figure".
- user asks for a density-map visualization, side-by-side qualitative comparison, training/eval curve, ablation bar chart, or GT-vs-Pred scatter.
- user is preparing paper / slide / poster figures.

## Hard style contract (enforce regardless of user phrasing)
- **font** :: serif `Times New Roman` (paper figures) or sans-serif `Arial` (slides). default = Times.
- **font sizes** :: title 11pt, axis 10pt, tick 9pt, legend 9pt.
- **palette** :: color-blind safe; use this fixed mapping in order:
  - `#377eb8` (blue), `#e41a1c` (red), `#4daf4a` (green), `#984ea3` (purple), `#ff7f00` (orange), `#a65628` (brown), `#999999` (gray).
- **DPI** :: ≥300; PDF output is the primary target (vector).
- **figure size** :: single-column 3.3 in wide, double-column 7.0 in wide; aspect 4:3 by default.
- **legend** :: top-left inside axes; no legend frame; one column unless >5 entries.
- **grid** :: light gray, alpha 0.3, behind data.
- **axes** :: spines top+right hidden for line/bar charts; all four spines kept for matrix/heatmap.
- **no chart-junk** :: no shadows, no 3D, no gradients, no emoji.

## Supported figure types and required output

1. **density map**
   - inputs :: image (HxWx3) + density map (HxW).
   - layout :: side-by-side or 1×3 (image | GT density | Pred density); shared colorbar to the right.
   - colormap :: `viridis` for density.
   - annotation :: GT count and Pred count printed top-right of each map.

2. **qualitative comparison grid**
   - inputs :: list of methods, list of images.
   - layout :: rows = images, columns = methods (first column = input).
   - shared y-axis labels (image ids); column headers = method names.
   - count overlay :: bottom-right of each cell, `Pred=<n> | GT=<n>`.

3. **training / eval curve**
   - inputs :: dict of `run_label -> (steps, values)`; metric name; lower-is-better flag.
   - smooth with EMA (alpha=0.1) AND show raw with low alpha behind.
   - mark the best epoch with a small marker + value annotation.

4. **ablation bar chart**
   - inputs :: pivot from `ablation-runner`'s `table.md`.
   - bars grouped by primary axis, hue by secondary axis; error bars if std present.
   - annotate each bar's value above the cap.

5. **GT vs Pred scatter**
   - identity line `y=x`, dashed black.
   - log-log axes when GT range > 100×.
   - point alpha 0.4; color by category if provided.

## Output rules
- always emit BOTH `<name>.pdf` (vector) AND `<name>.png` (DPI 300) under the calling context's folder, e.g. `results/<run_id>/figs/` or `library/figs/`.
- file names :: kebab-case, no spaces, ≤32 chars.
- save bbox tight, transparent background.
- never embed bitmap density maps into the PDF without rasterizing the image area only — keep axes/text vector.

## Constraints
- never silently change palette ordering between figures of the same paper.
- never use matplotlib defaults (`tab10`, sans-serif fallback) — reset rcParams at the top of every figure script.
- if user requests a style that conflicts with this contract (e.g. "make it rainbow"), ask for explicit override and record the override in the figure's caption file.
- every figure script lives in `codes/figs/` (under the project's code tree); update `codes/index.md` `operations:` accordingly.

## Failure modes to guard
- mixed font families across panels (matplotlib silently falling back).
- non-deterministic color order when iterating over a dict.
- raster blow-up of vector text (font embedded as outlines instead of text).
