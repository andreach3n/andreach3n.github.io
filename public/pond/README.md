# Pond art

Drop your hand-drawn images here. Because this is inside `public/`, a file at
`public/pond/whatever.png` is served on the site at `/pond/whatever.png`.

## What goes where

- **Lilypads** → `public/pond/lilypads/` (e.g. `pad1.png`, `pad2.png`, …)
  Then point a project at it in `src/pages/projects.astro` with `img: '/pond/lilypads/pad1.png'`.
- **Background** → `public/pond/pond-bg.png`
  The projects page automatically uses this as the pond background if the file exists.
- **Frog / critters** (later) → `public/pond/frog.png`, etc.

## Tips for the drawings

- **Format:** PNG with a **transparent background** for lilypads and the frog
  (so the pond shows around their edges). SVG works too if you draw vector art.
- **Background:** PNG or JPG is fine (no transparency needed).
- **Resolution:** draw at ~2× the display size for crispness on retina screens.
  A pad shows around 190px, so ~380–400px art is plenty. The background shows
  up to ~800px wide, so ~1600px wide is a good target.
