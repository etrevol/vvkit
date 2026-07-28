# Brand assets

| File | Size | Purpose |
|---|---|---|
| `mascot.svg` | 240×240 | Square logo / avatar / favicon source |
| `wordmark.svg` | 432×160 | Wordmark, auto light/dark via `prefers-color-scheme` |
| `wordmark-light.svg` / `wordmark-dark.svg` | 432×160 | Explicit variants for `<picture>` |
| `banner.svg` | 1200×400 | README header |
| `social-preview.png` | 1280×640 | GitHub → Settings → Social preview |

Palette: `#ff1493` `#ff69b4` `#ffffff` `#1a0510` `#000000`. Grid: 16 px.

## README header

```html
<p align="center">
  <img src="brand/banner.svg" alt="vvkit" width="100%">
</p>
```

## Logo + wordmark separately

```html
<p align="center">
  <img src="brand/mascot.svg" alt="" width="96"><br>
  <img src="brand/wordmark.svg" alt="vvkit" width="240">
</p>
```

`wordmark.svg` carries its own `@media (prefers-color-scheme: dark)` rule, so it
switches ink automatically. If a renderer strips inline `<style>`, use the
explicit variants instead:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/wordmark-dark.svg">
  <img src="brand/wordmark-light.svg" alt="vvkit" width="240">
</picture>
```

## Notes

- SVGs have no `width`/`height` attributes, only `viewBox` — the consumer sets
  the size. Each has `role="img"` and a `<title>` for screen readers.
- Clear space around the lockup is 16 px per 208 px of mascot height (1 grid cell).
- `shape-rendering="crispEdges"` sits on the root element, so pixels stay sharp
  at any scale.
