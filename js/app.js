// Raksha Bandhan — virtual rakhi camera app
//
// Flow: user taps "Open Camera" -> back (rear) camera opens -> an
// instructional banner slides down asking to show a hand -> MediaPipe
// HandLandmarker tracks hands in the live video -> once a hand is
// confidently and steadily detected, the banner slides back up and out, and
// the animated rakhi SVG (assets/rakhi.svg) is live-injected into the DOM
// at the wrist, sized/rotated every frame so it never outgrows the wrist.
// The whole thing plays exactly once per camera session: the rakhi ties
// on, holds, fades out, and a plain "Happy Raksha Bandhan" message floats
// into the center of the screen to close it out — no looping, no
// re-triggering while the hand stays in view.

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

const RAKHI_SVG_URL = new URL("../assets/rakhi.svg", import.meta.url).href;

// assets/rakhi.svg's native viewBox, and the pendant's circle within it
// (measured from the artwork: the round mandala sits slightly right of
// center, with loose thread tails trailing off both sides). Sizing the
// element from the pendant's true diameter — not the whole viewBox, which
// includes the dangling thread tails — is what keeps the pendant itself
// snug on the wrist while letting the tails drape past it naturally, the
// way a real tied rakhi does.
const SVG_VB_W = 360;
const SVG_VB_H = 200;
const PENDANT_CENTER = { x: 201, y: 96 };
const PENDANT_DIAMETER = 104;

// Hand landmark indices (MediaPipe Hand Landmarker topology).
const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

// Detection tuning.
const MIN_HAND_SCORE = 0.5;
const CONFIRM_FRAMES = 5; // consecutive frames of a steady hand before we commit
const SMOOTH = 0.35; // exponential-smoothing weight for the new sample each frame

// A wrist is narrower than the hand's knuckle row, and sits a little
// further up the arm than the WRIST landmark itself. These ratios (against
// the index-to-pinky knuckle span) approximate both.
const WRIST_WIDTH_RATIO = 0.62;
const WRIST_OFFSET_RATIO = 0.18;

// One-shot sequence timing: assets/rakhi.svg's own tie-on animation is
// authored to run for 4.033s (see the loop fix in loadRakhiMarkup below),
// after which we hold the fully-tied rakhi on screen for a beat, fade it
// out, and reveal the closing message. FADE_MS must match the CSS
// transition duration on .rakhi-mount.fade-out, and the .final-message
// entrance transition, in style.css.
const TIE_ON_MS = 4033;
const HOLD_AFTER_TIE_MS = 500;
const FADE_MS = 350;

// ---------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------

// The start screen IS the "Open Camera" button — a plain full-screen tap
// target, no card/title/copy — so both names point at the same element.
const startScreenEl = document.getElementById("start-screen");
const cameraScreenEl = document.getElementById("camera-screen");
const openCameraBtn = startScreenEl;

const videoStageEl = document.getElementById("video-stage");
const videoEl = document.getElementById("camera-feed");
const rakhiMountEl = document.getElementById("rakhi-mount");
const instructionBannerEl = document.getElementById("instruction-banner");
const finalMessageEl = document.getElementById("final-message");

const loadingEl = document.getElementById("loading-indicator");
const loadingTextEl = document.getElementById("loading-text");
const errorBoxEl = document.getElementById("error-box");
const errorTextEl = document.getElementById("error-text");
const errorDetailEl = document.getElementById("error-detail");
const errorRetryBtn = document.getElementById("error-retry");

const flashToggleBtn = document.getElementById("flash-toggle");
const flashGlowEl = document.getElementById("flash-glow");

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

let stream = null;
let handLandmarker = null;
let rakhiSvgMarkup = null;

let torchSupported = false;
let torchOn = false;
let flashSimOn = false;
let isMirrored = false; // true only if the browser ends up giving us a front camera

/** @type {"waiting" | "banded" | "complete"} */
let trackState = "waiting";
let handStreak = 0;
let smoothed = null; // { x, y, angle, width } in on-screen pixels
let rafId = null;
let lastVideoTime = -1;
let sequenceTimers = []; // pending setTimeout ids for the fade/finish steps

// ---------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------

openCameraBtn.addEventListener("click", begin);
errorRetryBtn.addEventListener("click", begin);
flashToggleBtn.addEventListener("click", toggleFlash);

async function begin() {
  showScreen("camera");
  hideError();
  showLoading("Requesting camera access…");

  // Reset flash UI left over from a previous attempt — otherwise a retry
  // after an error can show the flash icon as "on" even though the stream
  // (and any real torch) it referred to is already gone.
  flashToggleBtn.setAttribute("aria-pressed", "false");
  flashGlowEl.classList.remove("on");
  flashSimOn = false;
  torchOn = false;

  // Tracks which step we were on when something throws, so the error
  // message (and the technical detail shown under it) says what actually
  // failed instead of always blaming "the camera" for a model/asset issue.
  let stage = "camera";

  try {
    await startCamera();
    stage = "tracking";
    showLoading("Loading hand tracking…");
    await Promise.all([loadModel(), loadRakhiMarkup()]);

    hideLoading();
    clearSequenceTimers();
    trackState = "waiting";
    handStreak = 0;
    smoothed = null;
    hideRakhi();
    rakhiMountEl.classList.remove("fade-out");
    finalMessageEl.classList.remove("show");
    showBanner();
    startDetectionLoop();
  } catch (err) {
    console.error(`Failed to start (stage: ${stage}):`, err);
    stopCameraStream();
    hideLoading();
    showError(describeError(err, stage), describeErrorDetail(err));
  }
}

// ---------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("unsupported-camera");
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { exact: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  }).catch(async (err) => {
    // Some laptops/desktops (and a few devices) have no rear camera at all,
    // so `exact: "environment"` throws OverconstrainedError. Fall back to
    // whatever camera is available rather than dead-ending the whole flow.
    if (!err || err.name !== "OverconstrainedError") throw err;
    return navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  });

  videoEl.srcObject = stream;
  await new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = () => resolve();
    videoEl.onerror = () => reject(new Error("video-error"));
  });
  await videoEl.play();

  const track = stream.getVideoTracks()[0];
  // getSettings/getCapabilities are widely supported but not guaranteed
  // everywhere (and getCapabilities can itself throw on some browsers
  // rather than just being undefined) — read them defensively so a quirk
  // here can never take down camera startup entirely.
  let settings = {};
  let caps = {};
  try {
    settings = track.getSettings ? track.getSettings() : {};
  } catch (err) {
    console.error("track.getSettings() failed:", err);
  }
  try {
    caps = track.getCapabilities ? track.getCapabilities() : {};
  } catch (err) {
    console.error("track.getCapabilities() failed:", err);
  }
  // The rear camera is not mirrored on screen — it isn't a selfie view, it
  // shows the scene as the camera actually sees it, same as a photo.
  isMirrored = settings.facingMode === "user";
  videoStageEl.classList.toggle("mirrored", isMirrored);
  // Rear cameras usually expose a real hardware torch (unlike front ones);
  // when one doesn't, the flash toggle below falls back to a screen-flash.
  torchSupported = !!caps.torch;
  torchOn = false;
}

function stopCameraStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  videoEl.srcObject = null;
}

// ---------------------------------------------------------------------
// Flash / torch toggle (top bar)
// ---------------------------------------------------------------------

async function toggleFlash() {
  if (!stream) return;
  const track = stream.getVideoTracks()[0];

  if (torchSupported) {
    try {
      const next = !torchOn;
      await track.applyConstraints({ advanced: [{ torch: next }] });
      torchOn = next;
      flashToggleBtn.setAttribute("aria-pressed", String(torchOn));
      return;
    } catch (err) {
      // Some browsers report the capability but reject the constraint —
      // fall back to the simulated flash for the rest of the session.
      torchSupported = false;
    }
  }

  flashSimOn = !flashSimOn;
  flashGlowEl.classList.toggle("on", flashSimOn);
  flashToggleBtn.setAttribute("aria-pressed", String(flashSimOn));
}

// ---------------------------------------------------------------------
// Model + asset loading
// ---------------------------------------------------------------------

async function loadModel() {
  if (handLandmarker) return;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const options = {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: MIN_HAND_SCORE,
    minHandPresenceConfidence: MIN_HAND_SCORE,
    minTrackingConfidence: MIN_HAND_SCORE,
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, options);
  } catch (err) {
    // GPU delegate isn't available everywhere — retry on CPU.
    options.baseOptions.delegate = "CPU";
    handLandmarker = await HandLandmarker.createFromOptions(vision, options);
  }
}

async function loadRakhiMarkup() {
  if (rakhiSvgMarkup) return;
  const res = await fetch(RAKHI_SVG_URL);
  if (!res.ok) throw new Error("rakhi-asset");
  const raw = await res.text();

  // The artwork's tie-on animation is authored to loop forever
  // (repeatCount="indefinite" on every <animate>/<animateTransform>, all
  // sharing one 4.033s cycle) — great for a standalone sticker, but not for
  // a one-shot "ties on once" moment. Capping every repeat count at 1 stops
  // the loop. The visibility toggles that reveal each stroke as it "draws
  // in" have no fill="freeze" (unlike the pop-in scale animations, which
  // already do), so left alone they'd snap back to hidden the instant the
  // single rep finished — freezing them too keeps the whole assembled
  // rakhi visible together once the animation completes, so *our* fade-out
  // below is what actually makes it disappear, on our own timing.
  rakhiSvgMarkup = raw
    .replace(/repeatCount="indefinite"/g, 'repeatCount="1"')
    .replace(/attributeName="visibility" \/>/g, 'attributeName="visibility" fill="freeze" />');
}

// ---------------------------------------------------------------------
// Detection loop
// ---------------------------------------------------------------------

function startDetectionLoop() {
  if (rafId) cancelAnimationFrame(rafId);

  const loop = () => {
    // Once the one-shot sequence has played out, there's nothing left to
    // track — stop scheduling frames entirely rather than idling forever.
    if (trackState === "complete") {
      rafId = null;
      return;
    }
    rafId = requestAnimationFrame(loop);

    if (!handLandmarker || videoEl.readyState < 2) return;
    if (videoEl.currentTime === lastVideoTime) return; // no new frame yet
    lastVideoTime = videoEl.currentTime;

    const results = handLandmarker.detectForVideo(videoEl, performance.now());
    const hand = pickBestHand(results);

    if (trackState === "waiting") {
      handStreak = hand ? handStreak + 1 : 0;
      if (handStreak >= CONFIRM_FRAMES) {
        commitRakhi();
      }
      return;
    }

    // trackState === "banded": already committed to the one-shot sequence,
    // which is now running on its own timer (see commitRakhi) regardless of
    // whether we keep seeing the hand — a brief tracking hiccup mid-animation
    // no longer cancels or restarts anything. Keep following the wrist for
    // as long as we can see it, purely for a smoother visual.
    if (hand) {
      updateRakhiPlacement(hand.landmarks);
    }
  };

  rafId = requestAnimationFrame(loop);
}

function clearSequenceTimers() {
  sequenceTimers.forEach(clearTimeout);
  sequenceTimers = [];
}

/**
 * Commits to placing the rakhi: shows it once (playing its tie-on
 * animation exactly once, see loadRakhiMarkup), holds it fully assembled
 * for a beat, fades it out, then reveals the closing message. Both steps
 * are scheduled up front, independently of each other, rather than
 * chained (one timer's callback starting the next) — so a hiccup in one
 * can never silently strand the other and leave the closing message
 * stuck unshown.
 */
function commitRakhi() {
  trackState = "banded";
  hideBanner();
  smoothed = null; // snap to the hand instead of easing in from nowhere
  showRakhi();

  clearSequenceTimers();
  const fadeAt = TIE_ON_MS + HOLD_AFTER_TIE_MS;
  sequenceTimers.push(
    setTimeout(() => {
      try {
        rakhiMountEl.classList.add("fade-out");
      } catch (err) {
        console.error("Failed to start rakhi fade-out:", err);
      }
    }, fadeAt),
    setTimeout(finishSequence, fadeAt + FADE_MS)
  );
}

/** Ends the one-shot sequence: rakhi is gone, closing message floats in. */
function finishSequence() {
  try {
    trackState = "complete";
    hideRakhi();
    rakhiMountEl.classList.remove("fade-out");
    finalMessageEl.classList.add("show");
  } catch (err) {
    console.error("Failed to finish the rakhi sequence:", err);
  }
}

/**
 * Finds the best hand to place the rakhi on in a HandLandmarker result.
 *
 * We ask for the right hand (matching Raksha Bandhan tradition), preferring
 * whichever detected hand appears to be it. But handedness relies on a
 * mirroring convention (MediaPipe's labelling assumes a mirrored/selfie
 * input; the raw frame we feed it here isn't, so the true anatomical hand
 * is the opposite of the reported label) that isn't equally reliable across
 * every browser/device — so a confidently-tracked hand is never left
 * un-detected just because that convention didn't hold: with no "right"
 * match, we fall back to the single most confident hand in frame.
 */
function pickBestHand(results) {
  if (!results || !results.landmarks || !results.landmarks.length) return null;
  const handednessList = results.handedness || results.handednesses || [];

  let best = null;
  let bestRight = null;
  for (let i = 0; i < results.landmarks.length; i++) {
    const category = handednessList[i] && handednessList[i][0];
    const score = category ? category.score : 0;
    if (score < MIN_HAND_SCORE) continue;

    const candidate = { landmarks: results.landmarks[i], score };
    if (!best || score > best.score) best = candidate;

    const realHand = category.categoryName === "Left" ? "Right" : "Left";
    if (realHand === "Right" && (!bestRight || score > bestRight.score)) {
      bestRight = candidate;
    }
  }
  return bestRight || best;
}

// ---------------------------------------------------------------------
// Wrist placement
// ---------------------------------------------------------------------

/** Maps a normalized (0..1) point in the raw camera frame to on-screen
 * pixels within the (object-fit: cover) video stage. */
function mapCoverPoint(normX, normY, videoW, videoH, stageW, stageH) {
  const scale = Math.max(stageW / videoW, stageH / videoH);
  const renderedW = videoW * scale;
  const renderedH = videoH * scale;
  const offsetX = (stageW - renderedW) / 2;
  const offsetY = (stageH - renderedH) / 2;
  return { x: offsetX + normX * renderedW, y: offsetY + normY * renderedH };
}

/** Shortest-path angle interpolation so the band doesn't spin the long way
 * around when the smoothed and target angles straddle the ±π wrap. */
function lerpAngle(current, target, t) {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  return current + diff * t;
}

function updateRakhiPlacement(landmarks) {
  const stageW = videoStageEl.clientWidth;
  const stageH = videoStageEl.clientHeight;
  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  if (!stageW || !stageH || !videoW || !videoH) return;

  const toStagePx = (pt) => mapCoverPoint(pt.x, pt.y, videoW, videoH, stageW, stageH);

  const wristPx = toStagePx(landmarks[WRIST]);
  const middlePx = toStagePx(landmarks[MIDDLE_MCP]);
  const indexPx = toStagePx(landmarks[INDEX_MCP]);
  const pinkyPx = toStagePx(landmarks[PINKY_MCP]);

  // Direction pointing from the hand out through the wrist and into the
  // forearm — the band's long axis runs perpendicular to this.
  let dx = wristPx.x - middlePx.x;
  let dy = wristPx.y - middlePx.y;
  const dirLen = Math.hypot(dx, dy) || 1;
  dx /= dirLen;
  dy /= dirLen;
  const bandAngle = Math.atan2(dy, dx) + Math.PI / 2;

  const knuckleWidth = Math.hypot(indexPx.x - pinkyPx.x, indexPx.y - pinkyPx.y);
  const targetWidth = knuckleWidth * WRIST_WIDTH_RATIO;
  const targetX = wristPx.x + dx * knuckleWidth * WRIST_OFFSET_RATIO;
  const targetY = wristPx.y + dy * knuckleWidth * WRIST_OFFSET_RATIO;

  if (!smoothed) {
    smoothed = { x: targetX, y: targetY, angle: bandAngle, width: targetWidth };
  } else {
    smoothed.x += (targetX - smoothed.x) * SMOOTH;
    smoothed.y += (targetY - smoothed.y) * SMOOTH;
    smoothed.width += (targetWidth - smoothed.width) * SMOOTH;
    smoothed.angle = lerpAngle(smoothed.angle, bandAngle, SMOOTH);
  }

  renderRakhiAt(smoothed);
}

/**
 * Positions/scales/rotates #rakhi-mount so the pendant's true center lands
 * exactly on `width`-sized wrist, with rotation pivoting around that same
 * point — so the pendant stays centered on the wrist as the hand turns,
 * and never grows wider than the wrist itself.
 */
function renderRakhiAt({ x, y, angle, width }) {
  const scale = width / PENDANT_DIAMETER;
  const containerW = SVG_VB_W * scale;
  const containerH = SVG_VB_H * scale;
  const left = x - PENDANT_CENTER.x * scale;
  const top = y - PENDANT_CENTER.y * scale;
  const originXPct = (PENDANT_CENTER.x / SVG_VB_W) * 100;
  const originYPct = (PENDANT_CENTER.y / SVG_VB_H) * 100;

  rakhiMountEl.style.width = `${containerW}px`;
  rakhiMountEl.style.height = `${containerH}px`;
  rakhiMountEl.style.left = `${left}px`;
  rakhiMountEl.style.top = `${top}px`;
  rakhiMountEl.style.transformOrigin = `${originXPct}% ${originYPct}%`;
  rakhiMountEl.style.transform = `rotate(${angle}rad)`;
}

function showRakhi() {
  if (!rakhiSvgMarkup) return;
  // Inserting the markup starts the SVG's own (now one-shot, see
  // loadRakhiMarkup) "tying on" animation. commitRakhi only calls this
  // once per session, so it plays exactly once, never replaying.
  rakhiMountEl.innerHTML = rakhiSvgMarkup;
}

function hideRakhi() {
  rakhiMountEl.innerHTML = "";
}

// ---------------------------------------------------------------------
// Banner + screen + loading/error UI helpers
// ---------------------------------------------------------------------

function showBanner() {
  instructionBannerEl.classList.add("show");
}

function hideBanner() {
  instructionBannerEl.classList.remove("show");
}

function showScreen(name) {
  startScreenEl.classList.toggle("hidden", name !== "start");
  cameraScreenEl.classList.toggle("hidden", name !== "camera");
}

function showLoading(text) {
  loadingTextEl.textContent = text;
  loadingEl.classList.remove("hidden");
}

function hideLoading() {
  loadingEl.classList.add("hidden");
}

function showError(text, detail) {
  errorTextEl.textContent = text;
  errorDetailEl.textContent = detail || "";
  errorDetailEl.classList.toggle("hidden", !detail);
  errorBoxEl.classList.remove("hidden");
}

function hideError() {
  errorBoxEl.classList.add("hidden");
}

function describeError(err, stage) {
  const name = err && err.name;
  const message = err && err.message;

  if (message === "unsupported-camera") {
    return "This browser doesn't support camera access. Try a recent Chrome, Safari, or Edge.";
  }
  if (message === "rakhi-asset") {
    return "Couldn't load the rakhi artwork. Please refresh and try again.";
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera access was denied. Please allow the camera permission and try again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera was found on this device.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is already in use by another app. Close it and try again.";
  }
  if (name === "OverconstrainedError") {
    return "Your camera doesn't support the requested settings.";
  }
  if (name === "SecurityError") {
    return "Camera access requires a secure connection (HTTPS or localhost).";
  }
  // Nothing matched a known camera-permission case — say which step it
  // actually failed on (see the `stage` tracking in begin()) instead of
  // always blaming "the camera" for what might be a tracking/asset issue,
  // and show the raw error below (see describeErrorDetail) so this isn't a
  // dead end without access to the browser console.
  if (stage === "tracking") {
    return "Something went wrong while loading hand tracking. Please try again.";
  }
  return "Something went wrong while starting the camera. Please try again.";
}

/** Raw technical detail shown in small print under the friendly message,
 * so an unexpected error is diagnosable without opening devtools. */
function describeErrorDetail(err) {
  if (!err) return "";
  const name = err.name;
  const message = err.message;
  if (name && message) return `${name}: ${message}`;
  return name || message || String(err);
}
