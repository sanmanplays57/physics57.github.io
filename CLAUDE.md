# CLAUDE.md

Personal theoretical-physics website by a physics student. Static HTML/CSS/JS, no
build step. Content = summaries/notes on physics & math, "study with me" paper
walkthroughs, and curated resources.

## Live site & deploy

- **Live URL:** https://sanmanplays57.github.io/physics57.github.io/
  (It is a *project* site, so it serves under `/physics57.github.io/`. The bare
  `https://physics57.github.io/` 404s because no account named `physics57` exists —
  the repo is owned by `sanmanplays57`. To get the clean root URL, the repo would
  need to live under an org literally named `physics57`, or use a custom domain.)
- **Deploy:** push to `main` → GitHub Pages rebuilds automatically (~1–3 min).
  Commit only when asked; push only when asked. Commit to `main` directly (it's the
  deploy branch for this solo site).
- Git user: `sanmanplays57`. End commit messages with the `Co-Authored-By` line.

## Structure

- `index.html` — home page (full cosmic background, intro, section links).
- `things/<Subject>/` — subjects (Classical Mechanics, QM, GR, QFT, ...). Each has
  an `index.html` list; topic pages are usually minimal stubs (MathJax + "Coming
  soon..."). Some subjects have `high_school/undergraduate/postgraduate` splits.
- `study/` — paper walkthroughs.
- `resources/` — Resource Recommendations (Other Cool Websites, String Theory,
  Miscellaneous Resources which also holds Lecture Notes).
- `miscellaneous/` — Hot takes, Math Art, History of Physics, Mathematica tutorials.

## Design system

- **Fonts (Google):** Space Grotesk (headings/hero/links-in-headings), Spectral
  (body), Cormorant 300 italic (the floating equations). Accent link color
  `#79D4FF` (hover `#B7EBFF`). Dark cosmic theme throughout.
- **`sky.css`** (repo root) — shared "empty sky" background (deep gradient + two
  drifting tiny-star layers). Linked into every *plain* page via a relative path
  with the correct `../` depth. **When adding a new plain page, add**
  `<link rel="stylesheet" href="<../ per depth>sky.css">` before `</head>`.
  Skip pages that have their own background (they contain `bg-video`, `page-bg`,
  or `hero-glow`): home, the GR/EM/CM/UG `index.html`s, `time_travel_and_warp_drives.html`,
  and the QNEC study page.
- **Video-background pattern:** fullscreen `<video ... muted playsinline>`; add
  `loop` for looping (GR), play-once-then-freeze-last-frame via JS (EM, with Ken
  Burns zoom + glowing dust canvas), or a pre-baked forward+reverse boomerang file
  looped (CM, plus SVG-turbulence wobble on mouse).
- **Home-page floating equations:** ambient tiny handwritten physics equations in
  the *margins only* (spawn only when viewport > ~1180px, so never on mobile).
  Lava gradient + SVG `#rip` turbulence warp, blur-in / blur-out, ~1–3 per 10s,
  dissolve on hover. Add/edit equations in the `FORMS` array in the inline script
  at the bottom of `index.html`. Respects `prefers-reduced-motion`.

## Tooling & recipes

- **No system ffmpeg.** Use the one bundled with the `imageio_ffmpeg` pip package:
  `C:/Users/manas/AppData/Local/Python/pythoncore-3.14-64/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`
- **Compress a background video** (small, no audio, web-optimized):
  `ffmpeg -i in.mp4 -an -c:v libx264 -crf 27 -preset slow -pix_fmt yuv420p -movflags +faststart out.mp4`
  (CRF ~27–28 fine for darkened/blurred backgrounds.)
- **Boomerang (forward+reverse) in one pass:**
  `ffmpeg -i in.mp4 -filter_complex "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[out]" -map "[out]" -an -c:v libx264 -crf 27 -preset slow -pix_fmt yuv420p -movflags +faststart out.mp4`
- **AI upscaling** (RTX 5060 available): Real-ESRGAN `realesrgan-ncnn-vulkan`
  (download the portable zip from the xinntao/Real-ESRGAN GitHub release). Use
  `-g 1` to target the NVIDIA GPU (device 0 is the Intel iGPU). Workflow: extract
  frames → `realesrgan-ncnn-vulkan -n realesrgan-x4plus -s 4 -g 1` → downscale to
  target res → encode. x4plus (general) for non-anime content.

## Gotchas

- **Case sensitivity:** GitHub Pages (Linux) is case-sensitive; Windows is not. A
  link/`src` whose capitalization doesn't exactly match the file works locally but
  404s live. Match case exactly.
- **Bandwidth:** GitHub Pages soft limit ~100 GB/month, ~1 GB repo. Keep videos
  small; they dominate. Home page uses a pure-CSS black hole (no video).
- **Web fonts only** for anything visitors must see (Chiller etc. are local-only —
  the site uses Google fonts instead).
- **Exotic glyphs** (script/combining Unicode like q̇, β̇, 𝒢) render inconsistently
  across browsers/fonts.
- Feature-support notes: `background-clip:text`, `filter:url(#svg)` (flaky in
  Safari), `@view-transition` (Chromium only), `backdrop-filter`/`mask-image`
  (need `-webkit-`). All degrade gracefully.
