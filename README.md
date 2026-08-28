# Raksha Bandhan — Virtual Rakhi Camera App

A small static web app: tap a link, your **front camera** opens, a banner
animates in asking you to **show your right hand**, and once it's detected
the banner slides back out while an animated **rakhi** ties itself onto your
wrist — sized and rotated live so it always fits, and tracks the wrist as it
moves.

## How it works

1. **Open Camera** (`index.html`) — a tap/click gesture requests
   `getUserMedia({ video: { facingMode: 'user' } })`. The preview is
   mirrored with CSS (`transform: scaleX(-1)`) for a natural, mirror-like
   selfie view.
2. **Flash** — a toggle in the top bar. Front cameras almost never expose a
   hardware torch, so the app tries the real
   [`torch` capability](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackCapabilities/torch)
   first and transparently falls back to a simulated on-screen flash glow
   when it isn't supported.
3. **Motion text** — `#instruction-banner` sits off-screen
   (`translateY(-140%)`) and slides down when the camera is ready, then
   slides back up once a right hand is confirmed. Pure CSS transition, no
   JS animation loop needed.
4. **Hand tracking** — [MediaPipe Tasks Vision — HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
   (loaded from jsDelivr/Google's model CDN, no server component) runs on
   every video frame via `requestAnimationFrame`. Handedness is used to
   find the **right** hand specifically (see the "raw vs. mirrored frame"
   note in `js/app.js` — the model sees the un-mirrored camera frame, so its
   left/right label is swapped to get the true anatomical hand). A short
   streak of consecutive confident detections (and a longer streak of
   misses before letting go) keeps the state stable and flicker-free.
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

## Running it

Camera access requires a secure context, so it must be served over
**HTTPS** or from **localhost** — opening `index.html` directly via
`file://` will not work. Any static file server is enough, e.g.:

```bash
npx http-server .        # or: python3 -m http.server 8080
```

Then open the printed `http://localhost:...` URL on a device with a front
camera (works great on mobile browsers over HTTPS too, e.g. via a tunnel or
deployed to GitHub Pages).

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
