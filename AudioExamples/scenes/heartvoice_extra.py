"""
Heart-Voice Coupling - extra scene.

Same as scene 4 (heart pulse + voice amplitude, with audio), but the voice
trace is the detrended residual (attack snipped, cubic trend removed, the
same signal the Lissajous scenes use). Renders a separate MP4.

Optional text card: add a `## detrended` block to heartvoice_script.md and it
is shown before the scene. Without that block the scene starts directly.

Usage: python heartvoice_extra.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------- config
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
TMP = os.environ.get("HV_TMP_EXTRA", os.path.join(BASE, "_render_tmp_extra"))
os.makedirs(TMP, exist_ok=True)

W, H, FPS = 1920, 1080, 30
DPI = 100
SR = 16000

BG = "#060608"
WHITE = "#f2f2f4"
DIM = "#63636e"
FAINT = "#2c2c33"
HEART = "#ff5964"
FONT = "Lato"

plt.rcParams.update({
    "font.family": FONT,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def sp(s):
    return " ".join(s)


def load_script(path):
    blocks, cur = {}, None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                cur = line[3:].strip()
                blocks[cur] = dict(lines=[], sub=[], duration=4.0)
            elif cur is None or not line.strip() or line.startswith("# "):
                continue
            elif line.startswith("duration:"):
                blocks[cur]["duration"] = float(line.split(":", 1)[1])
            elif line.startswith("sub:"):
                blocks[cur]["sub"].append(line.split(":", 1)[1].strip())
            else:
                blocks[cur]["lines"].append(line.strip())
    return blocks


SCRIPT = load_script(os.path.join(BASE, "heartvoice_script.md"))


# ---------------------------------------------------------------- signals
# (copied from heartvoice_video.py so that script is not executed on import;
#  keep the two in sync if you change the preprocessing there)
def bp(x, fs, lo, hi, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def lp(x, fs, hi, order=2):
    b, a = butter(order, hi / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def zscore(x, ref=None):
    r = x if ref is None else ref
    return (x - np.nanmean(r)) / np.nanstd(r)


class View:
    pass


class Trial:
    def __init__(self, cond, wav_name, csv_name, label_voice):
        self.cond = cond
        self.label_voice = label_voice
        w, wsr = sf.read(os.path.join(BASE, cond, wav_name))
        assert wsr == SR
        self.wav = w / (np.max(np.abs(w)) + 1e-12)
        df = pd.read_csv(os.path.join(BASE, cond, csv_name))
        t = df["time_s"].values
        self.fs = 2500.0
        self.t = t - t[0]
        self.dur = self.t[-1]
        env = df["env"].values.astype(float)
        pec = df["pectoralis_major"].values.astype(float)
        pec = np.nan_to_num(pec, nan=np.nanmedian(pec))

        fs = self.fs
        pec_disp = lp(pec, fs, 6)
        env_disp = lp(env, fs, 10)

        thr = 0.08 * np.max(env_disp)
        on = np.argmax(env_disp > thr) / fs
        off = (len(env_disp) - np.argmax(env_disp[::-1] > thr)) / fs
        ramp = np.clip((self.t - on) / 0.6, 0, 1)
        ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)

        pec_b = bp(pec, fs, 0.7, 3.0)
        env_b = bp(env, fs, 0.7, 3.0) * ramp
        post = self.t > on + 0.8
        pec_b = zscore(pec_b, pec_b[post])
        env_b = zscore(env_b, env_b[post])

        steady = self.t > on + 0.5
        pec_disp = zscore(pec_disp, pec_disp[steady])
        env_disp = zscore(env_disp, env_disp[steady])

        pad = 0.15
        c = max(0, int((on - pad) * fs))
        self.t = self.t[c:] - self.t[c]
        self.dur = self.t[-1]
        self.onset = min(on, pad)
        self.off_local = off - on + self.onset
        self.wav = self.wav[max(0, int((on - pad) * SR)):]
        self.pec_disp = pec_disp[c:]
        self.env_disp = env_disp[c:]
        self.pec_b = pec_b[c:]
        self.env_b = env_b[c:]

        self.ds = 10
        self.td = self.t[:: self.ds]
        self.pec_d = self.pec_disp[:: self.ds]
        self.env_d = self.env_disp[:: self.ds]
        self.pxb = self.pec_b[:: self.ds]
        self.exb = self.env_b[:: self.ds]
        self.fsd = fs / self.ds

        pk, _ = find_peaks(self.pec_disp, distance=int(0.45 * fs), height=0.6)
        self.beat_times = self.t[pk]

        self.attack = 0.8
        s0 = int((self.onset + self.attack) * fs)
        end_t = self.off_local - 0.25
        s1 = int(end_t * fs)
        if s1 <= s0 + int(1.0 * fs):
            s1 = len(self.t)
        ts = self.t[s0:s1] - self.t[s0]
        env_seg = self.env_disp[s0:s1]
        coef = np.polyfit(ts, env_seg, 3)
        trend = np.polyval(coef, ts)
        res = env_seg - trend
        res_s = lp(res, fs, 3.0)
        self.proc = dict(ts=ts, env=env_seg, trend=trend, res=res)

        ex_f = np.full(len(self.t), np.nan)
        px_f = np.full(len(self.t), np.nan)
        ex_f[s0:s1] = zscore(res_s)
        px_f[s0:s1] = self.pec_b[s0:s1]
        v = View()
        v.td = self.td
        v.exb = ex_f[:: self.ds]
        v.pxb = px_f[:: self.ds]
        v.dur = self.dur
        v.onset = self.onset + self.attack
        v.wav = self.wav
        self.snip = v


VOC = Trial("vocalize", "p8_trial_31_Micdenoised.wav",
            "pp_8_trial_31_heartrate_aligned.csv", "VOICE")

# ---------------------------------------------------------------- pipeline
FFMPEG = "ffmpeg"
video_tmp = os.path.join(TMP, "video.mp4")
proc = subprocess.Popen(
    [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
     "-pix_fmt", "yuv420p", video_tmp],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

frame_count = 0
audio_events = []


def push(fig):
    global frame_count
    fig.canvas.draw()
    proc.stdin.write(fig.canvas.buffer_rgba())
    frame_count += 1


def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)


def add_audio(start_frame, data, gain=0.85):
    d = data.astype(np.float64) * gain
    nf = int(0.03 * SR)
    if len(d) > 2 * nf:
        d[:nf] *= np.linspace(0, 1, nf)
        d[-nf:] *= np.linspace(1, 0, nf)
    audio_events.append((start_frame, d))


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def glow_line(ax, color, lw=1.8, zorder=5):
    layers = []
    for w_, a_ in [(lw * 4.5, 0.08), (lw * 2.2, 0.22), (lw, 1.0)]:
        ln, = ax.plot([], [], color=color, lw=w_, alpha=a_,
                      zorder=zorder, solid_capstyle="round")
        layers.append(ln)
    return layers


def set_glow(layers, x, y):
    for ln in layers:
        ln.set_data(x, y)


def glow_dot(ax, color, s=70, zorder=8):
    d1 = ax.scatter([], [], s=s * 5.5, color=color, alpha=0.12, zorder=zorder, lw=0)
    d2 = ax.scatter([], [], s=s * 2.2, color=color, alpha=0.30, zorder=zorder, lw=0)
    d3 = ax.scatter([], [], s=s, color=color, alpha=1.0, zorder=zorder, lw=0)
    return (d1, d2, d3)


def set_dot(dots, x, y):
    if not (np.isfinite(x) and np.isfinite(y)):
        for d in dots:
            d.set_offsets(np.empty((0, 2)))
        return
    for d in dots:
        d.set_offsets([[x, y]])


# ---------------------------------------------------------------- scenes
def scene_card(lines, dur, sub=None, size=34, subsize=20):
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    n = len(lines)
    maxlen = max((len(s) for s in lines), default=1)
    if maxlen > 66:
        size = size * 66.0 / maxlen
    txts = []
    y0 = 0.5 + 0.045 * (n - 1) + (0.03 if sub else 0)
    for i, s in enumerate(lines):
        txts.append(ax.text(0.5, y0 - 0.09 * i, s, ha="center", va="center",
                            color=WHITE, fontsize=size, fontweight="light"))
    if sub:
        for j, s in enumerate(sub):
            txts.append(ax.text(0.5, y0 - 0.09 * n - 0.02 - 0.055 * j, s,
                                ha="center", va="center", color=DIM,
                                fontsize=subsize))
    nf = int(dur * FPS)
    fade = int(0.7 * FPS)
    for f in range(nf):
        a = ease(min(1.0, f / fade, (nf - 1 - f) / fade))
        for t_ in txts:
            t_.set_alpha(a)
        push(fig)
    plt.close(fig)


def scene_two_series_detrended(tr, voice_label):
    """scene 4 layout; bottom trace = detrended voice residual"""
    res = tr.snip.exb                     # NaN outside the analysis window
    # play only the analysis window: animation and sound start where the
    # detrended residual starts, and stop where it ends
    t0 = tr.snip.onset
    t1 = t0 + tr.proc["ts"][-1]
    i_start = int(np.searchsorted(tr.td, t0))
    fig = new_fig()
    ax1 = fig.add_axes([0.08, 0.52, 0.84, 0.30])
    ax2 = fig.add_axes([0.08, 0.14, 0.84, 0.30])
    for ax in (ax1, ax2):
        ax.axis("off")
        ax.set_xlim(t0, t1)
    ax1.set_ylim(-3.2, 3.6)
    ax2.set_ylim(-3.2, 3.6)

    fig.text(0.08, 0.845, sp("HEART PULSE"), color=HEART, fontsize=18)
    fig.text(0.08, 0.465, sp(voice_label + " AMPLITUDE"), color=WHITE, fontsize=18)
    fig.text(0.08, 0.44, "detrended  ·  deviations from the overall trend",
             color=DIM, fontsize=14, va="top")

    ax2.plot([t0, t1], [0, 0], color=FAINT, lw=1.0)

    beat_times = [bt for bt in tr.beat_times if t0 <= bt <= t1]
    beats1 = [ax1.axvline(bt, color=WHITE, lw=0.8, alpha=0) for bt in beat_times]
    beats2 = [ax2.axvline(bt, color=WHITE, lw=0.8, alpha=0) for bt in beat_times]

    l1 = glow_line(ax1, HEART, lw=1.9)
    l2 = glow_line(ax2, WHITE, lw=1.9)
    d1 = glow_dot(ax1, HEART, s=45)
    d2 = glow_dot(ax2, WHITE, s=45)

    nf = int(round((t1 - t0) * FPS))
    # same audio as scene 4, cut to the same window as the animation
    add_audio(frame_count, tr.wav[int(t0 * SR):int(t1 * SR)] * 0.9)
    for f in range(nf):
        tcur = t0 + f / FPS
        i = max(np.searchsorted(tr.td, tcur), i_start + 2)
        set_glow(l1, tr.td[i_start:i], tr.pec_d[i_start:i])
        set_glow(l2, tr.td[i_start:i], res[i_start:i])
        set_dot(d1, tr.td[i - 1], tr.pec_d[i - 1])
        set_dot(d2, tr.td[i - 1], res[i - 1])
        for bt, b1, b2 in zip(beat_times, beats1, beats2):
            if tcur >= bt:
                a = min(1.0, (tcur - bt) / 0.25) * 0.16
                b1.set_alpha(a); b2.set_alpha(a)
        push(fig)
    plt.close(fig)


# ---------------------------------------------------------------- render
print("rendering extra scene...")

if "detrended" in SCRIPT:
    b = SCRIPT["detrended"]
    scene_card(b["lines"], b["duration"], sub=b["sub"] or None)

scene_two_series_detrended(VOC, "VOICE")

proc.stdin.close()
proc.wait()
print(f"video frames: {frame_count} ({frame_count / FPS:.1f}s)")

# ---------------------------------------------------------------- audio mux
total = int(round(frame_count / FPS * SR))
track = np.zeros(total)
for start_frame, seg in audio_events:
    s0 = int(round(start_frame / FPS * SR))
    s1 = min(s0 + len(seg), total)
    track[s0:s1] += seg[: s1 - s0]
track = np.clip(track, -1, 1)
audio_tmp = os.path.join(TMP, "audio.wav")
sf.write(audio_tmp, track, SR)

out = os.path.join(OUT_DIR, "heartvoice_extra_detrended.mp4")
subprocess.run([FFMPEG, "-y", "-i", video_tmp, "-i", audio_tmp,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", out],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("done:", out)