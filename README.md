# Raksha Bandhan — Virtual Rakhi Camera App

A small static web app: tap a link, your **back (rear) camera** opens, a
banner animates in asking you to **show your right hand**, and once it's
detected the banner slides back out while an animated **rakhi** ties itself
onto your wrist once — sized, rotated, and tracked live so it always fits —
then fades out and a **"Happy Raksha Bandhan"** message slides in to close
things out. The whole sequence plays exactly once per camera session: no
looping, no re-triggering while the hand stays in view.

**Live app:** https://rajkumar542.github.io/raksha_bandhan/ (deploys
automatically from `main` — see [Hosting](#hosting) below). Camera access
needs HTTPS, which GitHub Pages provides.

## How it works

1. **Open Camera** (`index.html`) — a tap/click gesture requests
   `getUserMedia({ video: { facingMode: { exact: 'environment' } } })` (the
   rear camera). If a device genuinely has no rear camera, that throws
   `OverconstrainedError` and the app transparently retries with whatever
   camera is available — mirroring the preview with CSS only in that
   fallback case, since a rear camera should show the scene as-is, like a
   normal photo, not like a mirror.
2. **Flash** — a toggle in the top bar. Rear cameras usually expose a real
   hardware torch, so the app tries the
   [`torch` capability](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackCapabilities/torch)
   first and transparently falls back to a simulated on-screen flash glow
   on the (rarer) devices where it isn't supported.
3. **Motion text** — `#instruction-banner` sits off-screen
   (`translateY(-140%)`) and slides down when the camera is ready, then
   slides back up once a hand is confirmed. Pure CSS transition, no JS
   animation loop needed. The same banner reappears once more at the end —
   restyled gold via a `.final` class — with the closing "Happy Raksha
   Bandhan" message.
4. **Hand tracking** — [MediaPipe Tasks Vision — HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
   (loaded from jsDelivr/Google's model CDN, no server component) runs on
   every video frame via `requestAnimationFrame`. Handedness is used to
   *prefer* whichever detected hand looks like the right hand (matching
   Raksha Bandhan tradition), but that relies on a mirroring convention
   (see the note above `pickBestHand` in `js/app.js`) that isn't equally
   reliable everywhere — so if no hand matches, the single most confident
   hand in frame is used instead, rather than leaving a clearly-visible
   hand undetected. A short streak of consecutive confident detections (and
   a longer streak of misses before letting go) keeps the state stable and
   flicker-free.
5. **Rakhi placement** — `assets/rakhi.svg` (the original animated artwork,
   tie-on animation included) is injected live into the DOM — not drawn to
   a `<canvas>` — so its built-in animation actually plays. Every frame,
   `js/app.js` computes the wrist position, orientation, and width from the
   hand landmarks (wrist, index/pinky knuckles, middle knuckle) and sets
   the mount element's size/position/`transform-origin`/rotation so that
   the **pendant's true center** — not the whole artwork including its
   dangling thread tails — lands on the wrist and never grows wider than
   it, while the thread tails drape past it naturally, the way a real tied
   rakhi looks.
6. **One shot, not a loop** — the artwork's own tie-on animation is authored
   to repeat forever (`repeatCount="indefinite"` throughout). `js/app.js`
   patches every occurrence down to `repeatCount="1"` at load time (and
   freezes the visibility toggles that reveal each stroke, which otherwise
   have no `fill="freeze"` and would snap back to hidden the instant the
   single rep finished) so the whole rakhi draws in once and holds, fully
   assembled. After a short pause, `.rakhi-mount.fade-out` fades it out on
   our own timing, and the closing message banner takes over — see the
   `TIE_ON_MS` / `HOLD_AFTER_TIE_MS` / `FADE_MS` constants to retime any of
   it.

## Running it

Camera access requires a secure context, so it must be served over
**HTTPS** or from **localhost** — opening `index.html` directly via
`file://` will not work. Any static file server is enough, e.g.:

```bash
npx http-server .        # or: python3 -m http.server 8080
```

Then open the printed `http://localhost:...` URL on a device with a rear
camera (works great on mobile browsers over HTTPS too, e.g. via a tunnel or
deployed to GitHub Pages).

## Hosting

[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
publishes the site to GitHub Pages automatically on every push to `main`
(and can be run manually from the Actions tab). It's a plain static site —
no build step — so the workflow just uploads the repo (minus `.git`/`.github`)
as the Pages artifact.

If this is the first time Pages is being deployed for this repository, open
**Settings → Pages** once and confirm the source is set to **GitHub
Actions** (the `configure-pages` step in the workflow sets this
automatically in most cases, but it's worth a quick check on the very first
run). After that, the site is live at:

```
https://rajkumar542.github.io/raksha_bandhan/
```

## Files

```
index.html        Markup for the start screen and camera screen
css/style.css      Layout, mirroring, banner motion, flash glow
js/app.js          Camera, MediaPipe hand tracking, wrist placement logic
assets/rakhi.svg   The animated rakhi artwork
```

## Notes / limitations

- Hand landmarks alone don't include the forearm, so the wrist's position
  and orientation are estimated from the palm's base and knuckle row —
  works well in practice but is an approximation, not exact anatomy.
- Requires a browser with `getUserMedia` and WebAssembly support (all
  current Chrome, Safari, Edge, Firefox).
- Nothing is recorded, stored, or uploaded — the camera stream and all hand
  tracking run entirely client-side in the browser.
