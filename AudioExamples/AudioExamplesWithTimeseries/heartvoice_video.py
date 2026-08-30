"""
Heart-Voice Coupling — animated explainer video.

Builds a single MP4 (1920x1080, 30 fps, with audio) from the aligned
audio + physiology CSVs. Black theme, white/soft-red lines, Lato type.

Scenes:
  intro card -> heart pulse (silent) -> heart pulse + voice audio ->
  two timeseries + audio -> Lissajous construction (heart x, voice y) ->
  rotate 90 deg + replay (voice x) -> slow motion 0.5x (pitch-dropped audio) ->
  expiration timeseries + Lissajous -> reference card.

Usage: python heartvoice_video.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------- config
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
TMP = os.environ.get("HV_TMP", os.path.join(BASE, "_render_tmp"))
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
    """letterspaced label"""
    return " ".join(s)


def load_script(path):
    """parse heartvoice_script.md into {id: {lines, sub, duration}}"""
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
    """lightweight signal bundle with the same fields scene_lissajous needs"""
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

        # onset of sound: smooth ramp so bandpass transient doesn't blow up
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

        # scale display traces on the voiced part only: including the silent
        # lead-in compresses the hills and valleys into a flat band
        steady = self.t > on + 0.5
        pec_disp = zscore(pec_disp, pec_disp[steady])
        env_disp = zscore(env_disp, env_disp[steady])

        # crop everything (signals + audio) to start at vocalization
        pad = 0.15
        c = max(0, int((on - pad) * fs))
        self.t = self.t[c:] - self.t[c]
        self.dur = self.t[-1]
        self.onset = min(on, pad)
        self.off_local = off - on + self.onset   # voicing offset, cropped time
        self.wav = self.wav[max(0, int((on - pad) * SR)):]
        self.pec_disp = pec_disp[c:]
        self.env_disp = env_disp[c:]
        self.pec_b = pec_b[c:]
        self.env_b = env_b[c:]

        # display downsample (250 Hz)
        self.ds = 10
        self.td = self.t[:: self.ds]
        self.pec_d = self.pec_disp[:: self.ds]
        self.env_d = self.env_disp[:: self.ds]
        self.pxb = self.pec_b[:: self.ds]
        self.exb = self.env_b[:: self.ds]
        self.fsd = fs / self.ds

        pk, _ = find_peaks(self.pec_disp, distance=int(0.45 * fs), height=0.6)
        self.beat_times = self.t[pk]

        # paper procedure: snip the attack, detrend the envelope, keep the
        # residuals. The Lissajous scenes use this snipped view.
        self.attack = 0.8
        s0 = int((self.onset + self.attack) * fs)
        end_t = self.off_local - 0.25            # trim the offset plunge too
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

        # full-timeline view: audio plays the whole vocalization (same length
        # as the first playback); the residuals are NaN outside the analysis
        # window, so the Lissajous only draws where the procedure applies
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
VOC2 = Trial("vocalize", "p8_trial_18_Micdenoised.wav",
             "pp_8_trial_18_heartrate_aligned.csv", "VOICE")
EXP = Trial("expire", "p8_trial_12_Micdenoised.wav",
            "pp_8_trial_12_heartrate_aligned.csv", "BREATH")

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
audio_events = []  # (start_frame, np.array)


def push(fig):
    global frame_count
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    proc.stdin.write(buf)
    if frame_count % 200 == 0:
        import PIL.Image
        PIL.Image.frombuffer("RGBA", (W, H), bytes(buf)).save(
            os.path.join(TMP, f"sample_{frame_count:05d}.png"))
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
    """three stacked lines -> soft glow; returns list, use set_data on each"""
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
    for d in dots:
        d.set_offsets([[x, y]])


def hide_dot(dots):
    for d in dots:
        d.set_offsets(np.empty((0, 2)))


# ---------------------------------------------------------------- scenes
def scene_card(lines, dur, sub=None, size=34, subsize=20):
    """centered fading text card. lines: list of str"""
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    n = len(lines)
    maxlen = max((len(s) for s in lines), default=1)
    if maxlen > 66:                      # keep long script lines on screen
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
        a = min(1.0, f / fade, (nf - 1 - f) / fade)
        a = ease(a)
        for t_ in txts:
            t_.set_alpha(a)
        push(fig)
    plt.close(fig)


def scene_pulse(tr, with_audio, extra_label=None):
    """full-width heart pulse drawing left to right"""
    fig = new_fig()
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.60])
    ax.axis("off")
    ax.set_xlim(0, tr.dur)
    ax.set_ylim(-3.2, 3.6)

    fig.text(0.07, 0.86, sp("HEART PULSE"), color=WHITE, fontsize=21)
    fig.text(0.07, 0.815, "recorded at the chest  ·  pectoralis major",
             color=DIM, fontsize=15)
    if extra_label:
        fig.text(0.5, 0.09, extra_label, color=DIM, fontsize=16, ha="center")

    line = glow_line(ax, HEART, lw=2.0)
    dots = glow_dot(ax, HEART, s=55)
    base, = ax.plot([0, tr.dur], [-3.0, -3.0], color=FAINT, lw=1)
    prog = glow_line(ax, DIM, lw=1.2, zorder=3)

    nf = int(round(tr.dur * FPS))
    if with_audio:
        add_audio(frame_count, tr.wav * 0.9)
    for f in range(nf):
        tcur = f / FPS
        i = np.searchsorted(tr.td, tcur)
        i = max(i, 2)
        set_glow(line, tr.td[:i], tr.pec_d[:i])
        set_dot(dots, tr.td[i - 1], tr.pec_d[i - 1])
        set_glow(prog, [0, tcur], [-3.0, -3.0])
        push(fig)
    plt.close(fig)


def scene_two_series(tr, voice_label):
    """stacked heart + amplitude traces with beat alignment lines"""
    fig = new_fig()
    ax1 = fig.add_axes([0.08, 0.52, 0.84, 0.30])
    ax2 = fig.add_axes([0.08, 0.14, 0.84, 0.30])
    for ax in (ax1, ax2):
        ax.axis("off")
        ax.set_xlim(0, tr.dur)
    ax1.set_ylim(-3.2, 3.6)
    ax2.set_ylim(-3.2, 3.6)

    fig.text(0.08, 0.845, sp("HEART PULSE"), color=HEART, fontsize=18)
    fig.text(0.08, 0.465, sp(voice_label + " AMPLITUDE"), color=WHITE, fontsize=18)

    beats1 = [ax1.axvline(bt, color=WHITE, lw=0.8, alpha=0) for bt in tr.beat_times]
    beats2 = [ax2.axvline(bt, color=WHITE, lw=0.8, alpha=0) for bt in tr.beat_times]

    l1 = glow_line(ax1, HEART, lw=1.9)
    l2 = glow_line(ax2, WHITE, lw=1.9)
    d1 = glow_dot(ax1, HEART, s=45)
    d2 = glow_dot(ax2, WHITE, s=45)

    nf = int(round(tr.dur * FPS))
    add_audio(frame_count, tr.wav * 0.9)
    for f in range(nf):
        tcur = f / FPS
        i = max(np.searchsorted(tr.td, tcur), 2)
        set_glow(l1, tr.td[:i], tr.pec_d[:i])
        set_glow(l2, tr.td[:i], tr.env_d[:i])
        set_dot(d1, tr.td[i - 1], tr.pec_d[i - 1])
        set_dot(d2, tr.td[i - 1], tr.env_d[i - 1])
        for bt, b1, b2 in zip(tr.beat_times, beats1, beats2):
            if tcur >= bt:
                a = min(1.0, (tcur - bt) / 0.25) * 0.16
                b1.set_alpha(a); b2.set_alpha(a)
        push(fig)
    plt.close(fig)


def _liss_layout(xlabel, ylabel, xcolor, ycolor):
    """main square + left/bottom marginal panels; returns fig, axes dict"""
    fig = new_fig()
    mh = 0.60                      # main height fraction
    mw = mh * H / W                # square in pixels
    mx, my = 0.42, 0.24
    ax = fig.add_axes([mx, my, mw, mh])
    axl = fig.add_axes([mx - 0.115, my, 0.105, mh])       # left: y-signal
    axb = fig.add_axes([mx, my - 0.155, mw, 0.145])       # bottom: x-signal
    LIM = 3.4
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    for s_ in ax.spines.values():
        s_.set_color(FAINT); s_.set_linewidth(1.0)
    ax.set_xticks([]); ax.set_yticks([])

    WIN = 2.6
    axl.set_xlim(-WIN, 0); axl.set_ylim(-LIM, LIM); axl.axis("off")
    axb.set_xlim(-LIM, LIM); axb.set_ylim(-WIN, 0); axb.axis("off")

    fig.text(mx + mw / 2, my - 0.155 - 0.045, sp(xlabel), color=xcolor,
             fontsize=17, ha="center")
    fig.text(mx - 0.115 - 0.018, my + mh / 2, sp(ylabel), color=ycolor,
             fontsize=17, va="center", rotation=90)
    return fig, dict(ax=ax, axl=axl, axb=axb, WIN=WIN, LIM=LIM)


def scene_lissajous(tr, xsig, ysig, xlabel, ylabel, xcolor, ycolor,
                    rate=1.0, ghost=None, title=None, rotate_intro=False):
    """
    xsig/ysig: downsampled band-passed signals (same length as tr.td).
    rate: playback rate (0.5 = slow motion).
    ghost: optional (gx, gy) full curve drawn faint from the start.
    rotate_intro: animate ghost rotating -90 deg before the replay starts.
    """
    fig, L = _liss_layout(xlabel, ylabel, xcolor, ycolor)
    ax, axl, axb, WIN = L["ax"], L["axl"], L["axb"], L["WIN"]

    if title:
        fig.text(0.5, 0.925, title, color=DIM, fontsize=18, ha="center")

    # trail only after voicing onset (pre-onset the signal sits at zero
    # while the heart moves -> spurious straight line through the centre)
    i0 = int(np.searchsorted(tr.td, tr.onset + 0.15))

    ghost_ln = None
    if ghost is not None:
        ghost = (ghost[0][i0:], ghost[1][i0:])
        ghost_ln, = ax.plot([], [], color=WHITE, lw=1.0, alpha=0.13, zorder=2)

    hist, = ax.plot([], [], color=WHITE, lw=1.1, alpha=0.30, zorder=3)
    comet = glow_line(ax, WHITE, lw=2.2, zorder=6)
    pdot = glow_dot(ax, WHITE, s=60)

    lyl = glow_line(axl, ycolor, lw=1.5)
    lxb = glow_line(axb, xcolor, lw=1.5)
    dyl = glow_dot(axl, ycolor, s=30)
    dxb = glow_dot(axb, xcolor, s=30)

    conx = Line2D([], [], color=WHITE, lw=0.9, alpha=0.30, ls=(0, (4, 4)),
                  transform=fig.transFigure, zorder=1)
    cony = Line2D([], [], color=WHITE, lw=0.9, alpha=0.30, ls=(0, (4, 4)),
                  transform=fig.transFigure, zorder=1)
    fig.add_artist(conx); fig.add_artist(cony)

    def figxy(a, x, y):
        return fig.transFigure.inverted().transform(a.transData.transform((x, y)))

    # --- optional rotation intro
    if rotate_intro and ghost is not None:
        gx, gy = ghost
        nrot = int(1.8 * FPS)
        for s_ in (comet + [hist]):
            s_.set_data([], [])
        for f in range(nrot):
            th = -np.pi / 2 * ease(f / (nrot - 1))
            rx = gx * np.cos(th) - gy * np.sin(th)
            ry = gx * np.sin(th) + gy * np.cos(th)
            ghost_ln.set_data(rx, ry)
            ghost_ln.set_alpha(0.35)
            push(fig)
        # settle to faint
        for f in range(int(0.4 * FPS)):
            ghost_ln.set_alpha(0.35 - 0.22 * ease(f / (0.4 * FPS - 1)))
            push(fig)
        gxr = gx * 0 + gy       # rotated -90: (x,y)->(y,-x)
        gyr = -gx
        ghost_ln.set_data(gxr, gyr)

    elif ghost is not None:
        ghost_ln.set_data(*ghost)

    nf = int(round(tr.dur / rate * FPS))
    add_audio(frame_count, _make_scene_audio(tr, rate))
    comet_dur = 0.8 / rate  # seconds of bright trail (data time scaled ok)
    for f in range(nf):
        tcur = (f / FPS) * rate
        i = max(np.searchsorted(tr.td, tcur), 2)
        j = max(np.searchsorted(tr.td, tcur - comet_dur * rate), i0)
        h0 = min(i0, i)
        hist.set_data(xsig[h0:i] if i > i0 else [], ysig[h0:i] if i > i0 else [])
        set_glow(comet, xsig[j:i] if i > j else [], ysig[j:i] if i > j else [])
        cx, cy = xsig[i - 1], ysig[i - 1]

        m = (tr.td > tcur - WIN) & (tr.td <= tcur)
        trel = tr.td[m] - tcur
        set_glow(lyl, trel, ysig[m])
        set_glow(lxb, xsig[m], trel)

        if np.isfinite(cx) and np.isfinite(cy):
            set_dot(pdot, cx, cy)
            set_dot(dyl, 0, cy)
            set_dot(dxb, cx, 0)
            p_main = figxy(ax, cx, cy)
            p_left = figxy(axl, 0, cy)
            p_bot = figxy(axb, cx, 0)
            cony.set_data([p_left[0], p_main[0]], [p_left[1], p_main[1]])
            conx.set_data([p_bot[0], p_main[0]], [p_bot[1], p_main[1]])
        else:
            hide_dot(pdot); hide_dot(dyl); hide_dot(dxb)
            conx.set_data([], []); cony.set_data([], [])
        push(fig)
    plt.close(fig)


def scene_procedure(tr):
    """show the paper's preprocessing: snip attack, detrend, keep residuals"""
    fig = new_fig()
    ax = fig.add_axes([0.08, 0.20, 0.84, 0.56])
    ax.axis("off")
    ax.set_xlim(0, tr.dur)
    ax.set_ylim(-3.6, 3.6)

    title = fig.text(0.08, 0.845, sp("VOICE AMPLITUDE"), color=WHITE, fontsize=18)
    sub = fig.text(0.08, 0.80, "", color=DIM, fontsize=15)

    ds = tr.ds
    tcut = tr.onset + tr.attack                 # in cropped display time
    end_t = tcut + tr.proc["ts"][-1]            # stop before the offset tail
    kept = tr.td <= end_t
    td_full = tr.td[kept]
    env_full = tr.env_d[kept]
    ts = tr.proc["ts"][::ds]
    env_s = tr.proc["env"][::ds]
    trend = tr.proc["trend"][::ds]
    res = tr.proc["res"][::ds]
    G = float(np.clip(1.0 / (np.std(res) + 1e-9), 1.0, 6.0))

    env_ln = glow_line(ax, WHITE, lw=1.9)
    trend_ln, = ax.plot([], [], color=HEART, lw=2.0, ls=(0, (5, 4)),
                        alpha=0.0, zorder=6)
    cut_ln, = ax.plot([], [], color=DIM, lw=1.2, alpha=0.0)
    shade = ax.axvspan(0, tcut, color=BG, alpha=0.0, zorder=7)
    cut_txt = ax.text(tcut / 2, 3.1, "attack removed", color=DIM,
                      fontsize=14, ha="center", alpha=0.0)
    trend_txt = ax.text(0, 0, "overall trend", color=HEART, fontsize=14,
                        ha="left", alpha=0.0)
    base_ln, = ax.plot([0, tr.dur], [0, 0], color=FAINT, lw=1.0, alpha=0.0)

    def frames(sec):
        return int(round(sec * FPS))

    # P1: draw the envelope
    n = frames(1.2)
    for f in range(n):
        i = max(2, int(len(td_full) * ease((f + 1) / n)))
        set_glow(env_ln, td_full[:i], env_full[:i])
        push(fig)
    # P2: mark and dim the attack
    n = frames(1.0)
    for f in range(n):
        a = ease((f + 1) / n)
        cut_ln.set_data([tcut, tcut], [-3.4, 3.4]); cut_ln.set_alpha(0.8 * a)
        shade.set_alpha(0.82 * a)
        cut_txt.set_alpha(a)
        push(fig)
    for _ in range(frames(0.6)):
        push(fig)
    # P3: slide the kept part to the origin
    n = frames(0.9)
    for f in range(n):
        u = ease((f + 1) / n)
        fade = max(0.0, 1.0 - 2.5 * u)
        cut_ln.set_alpha(0.8 * fade); cut_txt.set_alpha(fade)
        shade.set_alpha(0.82 * fade)
        set_glow(env_ln, ts + (1 - u) * tcut, env_s)
        push(fig)
    shade.set_alpha(0)
    # P4: draw the fitted trend
    n = frames(1.1)
    trend_ln.set_alpha(0.9)
    for f in range(n):
        i = max(2, int(len(ts) * ease((f + 1) / n)))
        trend_ln.set_data(ts[:i], trend[:i])
        if f == n - 1:
            trend_txt.set_position((ts[-1] * 0.72, trend[int(len(ts)*0.72)] + 0.55))
            trend_txt.set_alpha(1.0)
        push(fig)
    for _ in range(frames(0.6)):
        push(fig)
    # P5: subtract the trend
    n = frames(1.3)
    for f in range(n):
        u = ease((f + 1) / n)
        set_glow(env_ln, ts, env_s - u * trend)
        trend_ln.set_data(ts, (1 - u) * trend)
        trend_txt.set_alpha(1.0 - u)
        base_ln.set_alpha(u)
        push(fig)
    # P6: amplify the residuals
    sub.set_text("deviations from the trend  ·  used for the Lissajous")
    n = frames(0.9)
    for f in range(n):
        u = ease((f + 1) / n)
        g = 1 + u * (G - 1)
        set_glow(env_ln, ts, res * g)
        sub.set_alpha(u)
        push(fig)
    for _ in range(frames(1.0)):
        push(fig)
    plt.close(fig)


def _make_scene_audio(tr, rate):
    w = tr.wav * 0.9
    if rate == 1.0:
        return w
    n = int(len(w) / rate)
    idx = np.linspace(0, len(w) - 1, n)
    return np.interp(idx, np.arange(len(w)), w)  # pitch drops with slow-mo


def scene_end():
    b = SCRIPT["reference"]
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    txts = []
    for k, s in enumerate(b["lines"]):
        txts.append(ax.text(0.5, 0.62 - 0.065 * k, s, ha="center",
                            color=WHITE, fontsize=23,
                            fontstyle="italic" if k == 1 else "normal"))
    for j, s in enumerate(b["sub"]):
        txts.append(ax.text(0.5, 0.35 - 0.05 * j, s, ha="center",
                            color=DIM, fontsize=18))
    nf = int(b["duration"] * FPS)
    fade = int(0.9 * FPS)
    for f in range(nf):
        a = ease(min(1.0, f / fade, (nf - 1 - f) / fade))
        for t_ in txts:
            t_.set_alpha(a)
        push(fig)
    plt.close(fig)


# ---------------------------------------------------------------- storyboard
def card(cid):
    b = SCRIPT[cid]
    scene_card(b["lines"], b["duration"], sub=b["sub"] or None)


print("rendering...")

card("intro")

card("heart-pulse")
scene_pulse(VOC, with_audio=False)

card("listen")
scene_pulse(VOC, with_audio=True, extra_label="●   with voice")

card("did-you-hear")
scene_two_series(VOC, "VOICE")

card("procedure")
scene_procedure(VOC)

card("lissajous")
scene_lissajous(VOC.snip, VOC.snip.pxb, VOC.snip.exb,
                "HEART", "VOICE", HEART, WHITE)

card("rotate")
scene_lissajous(VOC.snip, VOC.snip.exb, -VOC.snip.pxb,
                "VOICE", "HEART", WHITE, HEART,
                ghost=(VOC.snip.pxb, VOC.snip.exb), rotate_intro=True)

card("slowmo")
scene_lissajous(VOC.snip, VOC.snip.exb, -VOC.snip.pxb,
                "VOICE", "HEART", WHITE, HEART,
                rate=0.5, ghost=(VOC.snip.exb, -VOC.snip.pxb),
                title=sp("SLOW MOTION  ×0.5"))

card("another-example")
scene_lissajous(VOC2.snip, VOC2.snip.exb, -VOC2.snip.pxb,
                "VOICE", "HEART", WHITE, HEART)

card("expire")
scene_two_series(EXP, "BREATH")
scene_lissajous(EXP.snip, EXP.snip.exb, -EXP.snip.pxb,
                "BREATH", "HEART", WHITE, HEART)

card("expire-note")
scene_end()

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

out = os.path.join(OUT_DIR, "heartvoice_coupling.mp4")
subprocess.run([FFMPEG, "-y", "-i", video_tmp, "-i", audio_tmp,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", out],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("done:", out)
