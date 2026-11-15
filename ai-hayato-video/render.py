#!/usr/bin/env python3
"""AIハヤト 1年振り返り動画レンダラー (1920x1080 / 30fps)

使い方:
    python3 render.py                # script.json を読んで out/hayato_review.mp4 を出力
    python3 render.py --voice v.wav  # 音声を差し込む（口パクは音量に連動）
    python3 render.py --preview 8    # 8秒地点の静止画を out/preview.png に出力
"""
import argparse, json, math, os, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
CYAN = (90, 200, 255)
CYAN2 = (40, 140, 255)

# ---------- timeline (秒) ----------
T_PART = 2.0      # 粒子が集まる
T_SPIN = 1.4      # 回転登場
T_TALK = 2.6      # 中央で挨拶
T_MOVE = 0.8      # 左下へ移動
SLIDE = 0.55      # 写真スライドイン
# エンディング（t_outro からの相対秒）
O_MOVE0, O_MOVE1 = 0.25, 1.05   # 中央へ戻る
O_TALK0 = 1.15                  # 締めのセリフ
O_TALK_LEN = 3.1
O_WAVE0 = O_TALK0 + O_TALK_LEN + 0.1   # バイバイ
O_WAVE = 1.4
O_SPIN0 = O_WAVE0 + O_WAVE      # 回転して粒子に
O_SPIN = 1.1
O_DIS0 = O_SPIN0 + 0.45         # 粒子化スタート
O_DIS = 1.6
T_OUTRO = O_DIS0 + O_DIS + 0.4
T_FADE = 0.5


def ease_out(t):
    t = min(max(t, 0), 1); return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = min(max(t, 0), 1); return 3 * t * t - 2 * t * t * t


def ease_back(t):
    t = min(max(t, 0), 1); c = 1.6; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def font(size):
    return ImageFont.truetype(FONT, size)


def text_bold(draw, xy, text, size, fill, anchor="la", stroke=None):
    f = font(size)
    if stroke:
        draw.text(xy, text, font=f, fill=fill, anchor=anchor, stroke_width=stroke[0], stroke_fill=stroke[1])
    draw.text(xy, text, font=f, fill=fill, anchor=anchor, stroke_width=1, stroke_fill=fill)


# ---------- assets ----------
CROP = (500, 15, 1140, 935)  # 全ポーズ共通の切り抜き枠（ポーズを切り替えても位置がずれない）


def load_chara():
    d = os.path.join(ROOT, "assets/chara")
    out = {}
    for f in os.listdir(d):
        im = Image.open(os.path.join(d, f)).convert("RGBA").crop(CROP)
        a = np.array(im)
        a[..., 3] = np.where(a[..., 3] < 40, 0, a[..., 3])  # 薄いゴミを除去
        out[f[:-4]] = Image.fromarray(a)
    return out


def make_wave_parts(im):
    """人差し指ポーズから「手」だけを切り出す（バイバイ用）"""
    a = np.array(im)
    alpha = a[..., 3] > 100
    bx0, by0, bx1, by1 = 40, 425, 190, 577   # 切り抜き座標系での手の範囲
    box = np.zeros_like(alpha); box[by0:by1, bx0:bx1] = True
    region = alpha & box
    m = np.zeros_like(alpha); m[505, 100] = True
    for _ in range(400):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        n &= region
        if (n == m).all(): break
        m = n
    for _ in range(3):  # 縁のアンチエイリアス分を含める
        n = m.copy(); n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n & box
    hand = a.copy(); hand[..., 3] = np.where(m, a[..., 3], 0)
    body = a.copy(); body[..., 3] = np.where(m, 0, a[..., 3])
    return Image.fromarray(body), Image.fromarray(hand), (105, 577)


def make_bg():
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    top, bot = np.array([3, 12, 32]), np.array([8, 34, 74])
    img = top[None, None] * (1 - y[..., None]) + bot[None, None] * y[..., None]
    img = img * np.ones((1, W, 1))
    vign = 1 - 0.45 * (x ** 2)[..., None] * np.ones((H, 1, 1))
    img = (img * vign).clip(0, 255).astype(np.uint8)
    bg = Image.fromarray(img).convert("RGBA")
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    for gx in range(0, W, 80):
        d.line([(gx, 0), (gx, H)], fill=(80, 160, 255, 14))
    for gy in range(0, H, 80):
        d.line([(0, gy), (W, gy)], fill=(80, 160, 255, 14))
    # 下部のパース床ライン
    for i in range(18):
        x0 = -800 + i * 200
        d.line([(W / 2, 640), (x0 * 1.6 + W / 2 - 400, H)], fill=(80, 170, 255, 22))
    bg.alpha_composite(g)
    return bg


def header_layer(cfg):
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    text_bold(d, (1850, 34), cfg["header"], 24, CYAN + (255,), anchor="rt")
    text_bold(d, (1850, 68), cfg["title"], 44, (255, 255, 255, 255), anchor="rt")
    d.line([(1850 - 560, 128), (1850, 128)], fill=CYAN + (120,), width=2)
    return L


PANEL = (560, 165, 1850, 855)


def panel_card(photo_path, month, name):
    x0, y0, x1, y1 = PANEL
    pw, ph = x1 - x0, y1 - y0
    card = Image.new("RGBA", (pw + 80, ph + 80), (0, 0, 0, 0))
    # 外側の光
    glow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle((40, 40, 40 + pw, 40 + ph), 26, fill=CYAN2 + (150,))
    card.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
    base = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    ImageDraw.Draw(base).rounded_rectangle((0, 0, pw - 1, ph - 1), 24, fill=(10, 28, 58, 235), outline=CYAN + (230,), width=3)
    # 写真: 全体が見えるように収め、余白は同じ写真のぼかしで埋める
    pad = 18
    iw, ih = pw - pad * 2, ph - pad * 2
    if os.path.exists(photo_path):
        ph_img = Image.open(photo_path).convert("RGB")
    else:  # 写真未着の仮枠
        ph_img = Image.new("RGB", (1600, 900), (30, 50, 90))
        dd = ImageDraw.Draw(ph_img)
        text_bold(dd, (800, 400), f"{month} {name}", 90, (200, 230, 255), anchor="mm")
        text_bold(dd, (800, 530), "（写真差し替え予定）", 56, (140, 180, 230), anchor="mm")
    sc_cover = max(iw / ph_img.width, ih / ph_img.height)
    blur = ph_img.resize((int(ph_img.width * sc_cover) + 1, int(ph_img.height * sc_cover) + 1))
    blur = blur.crop(((blur.width - iw) // 2, (blur.height - ih) // 2, (blur.width - iw) // 2 + iw, (blur.height - ih) // 2 + ih))
    blur = blur.filter(ImageFilter.GaussianBlur(24))
    blur = Image.eval(blur, lambda v: int(v * 0.55))
    sc = min(iw / ph_img.width, ih / ph_img.height)
    fg = ph_img.resize((int(ph_img.width * sc), int(ph_img.height * sc)), Image.LANCZOS)
    blur.paste(fg, ((iw - fg.width) // 2, (ih - fg.height) // 2))
    mask = Image.new("L", (iw, ih), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, iw - 1, ih - 1), 16, fill=255)
    base.paste(blur, (pad, pad), mask)
    card.alpha_composite(base, (40, 40))
    return card


def badge(month, name):
    f1, f2 = font(64), font(44)
    w1 = f1.getbbox(month)[2] + 56
    w2 = f2.getbbox(name)[2] + 50
    Wb, Hb = w1 + w2 + 20, 100
    b = Image.new("RGBA", (Wb + 40, Hb + 40), (0, 0, 0, 0))
    g = Image.new("RGBA", b.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle((20, 20, 20 + Wb, 20 + Hb), 50, fill=CYAN2 + (200,))
    b.alpha_composite(g.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(b)
    d.rounded_rectangle((20, 20, 20 + w1, 20 + Hb), 50, fill=(40, 150, 255, 255), outline=(255, 255, 255, 255), width=3)
    text_bold(d, (20 + w1 / 2, 20 + Hb / 2), month, 64, (255, 255, 255, 255), anchor="mm")
    d.rounded_rectangle((20 + w1 + 14, 30, 20 + Wb, 20 + Hb - 10), 40, fill=(255, 255, 255, 245))
    text_bold(d, (20 + w1 + 14 + (Wb - w1 - 14) / 2, 20 + Hb / 2), name, 44, (10, 40, 100, 255), anchor="mm")
    return b


def wrap(text, f, maxw):
    if "\n" in text:
        return [l for part in text.split("\n") for l in wrap(part, f, maxw)]
    lines, cur = [], ""
    for ch in text:
        if f.getbbox(cur + ch)[2] > maxw and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur: lines.append(cur)
    return lines


def bubble(text, box, tail=True):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    b = Image.new("RGBA", (bw + 60, bh + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(b)
    ox, oy = 40, 20
    d.rounded_rectangle((ox, oy, ox + bw, oy + bh), 22, fill=(8, 24, 52, 225), outline=CYAN + (220,), width=3)
    if tail:
        d.polygon([(ox + 2, oy + bh / 2 - 18), (ox - 30, oy + bh / 2 + 6), (ox + 2, oy + bh / 2 + 18)], fill=(8, 24, 52, 225))
        d.line([(ox + 1, oy + bh / 2 - 18), (ox - 30, oy + bh / 2 + 6), (ox + 1, oy + bh / 2 + 18)], fill=CYAN + (220,), width=3)
    size = 48
    f = font(size)
    lines = wrap(text, f, bw - 60)
    if len(lines) > 2:
        size = 38; f = font(size); lines = wrap(text, f, bw - 60)
    lh = size + 14
    ty = oy + bh / 2 - lh * len(lines) / 2 + lh / 2
    for i, ln in enumerate(lines):
        text_bold(d, (ox + bw / 2, ty + i * lh), ln, size, (255, 255, 255, 255), anchor="mm")
    return b, (x0 - ox, y0 - oy)


def end_card():
    x0, y0, x1, y1 = PANEL
    pw, ph = x1 - x0, y1 - y0
    c = Image.new("RGBA", (pw + 80, ph + 80), (0, 0, 0, 0))
    g = Image.new("RGBA", c.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle((40, 40, 40 + pw, 40 + ph), 26, fill=CYAN2 + (170,))
    c.alpha_composite(g.filter(ImageFilter.GaussianBlur(22)))
    d = ImageDraw.Draw(c)
    d.rounded_rectangle((40, 40, 40 + pw, 40 + ph), 24, fill=(10, 30, 66, 240), outline=CYAN + (240,), width=3)
    cx, cy = 40 + pw / 2, 40 + ph / 2
    text_bold(d, (cx, cy - 120), "THANK YOU 2026", 40, CYAN + (255,), anchor="mm")
    text_bold(d, (cx, cy - 20), "今年も1年", 86, (255, 255, 255, 255), anchor="mm", stroke=(6, (20, 90, 200, 255)))
    text_bold(d, (cx, cy + 90), "ありがとうございました！", 86, (255, 255, 255, 255), anchor="mm", stroke=(6, (20, 90, 200, 255)))
    return c


# ---------- particles ----------
class Particles:
    def __init__(self, chara_img, box, n=2600, seed=3):
        rng = np.random.default_rng(seed)
        x0, y0, w, h = box
        m = np.array(chara_img.resize((w, h)).getchannel("A")) > 128
        ys, xs = np.nonzero(m)
        idx = rng.choice(len(xs), n)
        self.tgt = np.stack([xs[idx] + x0, ys[idx] + y0], 1).astype(float)
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = rng.uniform(500, 1300, n)
        cen = np.array([W / 2, H / 2])
        self.src = cen + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        self.delay = rng.uniform(0, 0.45, n)
        self.size = rng.choice([1.2, 1.8, 2.6], n, p=[0.5, 0.35, 0.15])
        self.col = np.array([[150, 220, 255], [70, 160, 255], [30, 110, 255], [220, 245, 255]])[rng.integers(0, 4, n)]
        self.burst = rng.normal(0, 1, (n, 2)) * 160
        self.cen = np.array([x0 + w / 2, y0 + h / 2])

    def draw(self, t):
        """t: 0..T_PART+0.8 秒"""
        lay = Image.new("RGB", (W // 2, H // 2), (0, 0, 0))
        d = ImageDraw.Draw(lay)
        if t <= T_PART:
            p = np.clip((t - self.delay) / (T_PART - 0.45), 0, 1)
            e = ease_in_out(p) if np.isscalar(p) else 3 * p * p - 2 * p ** 3
            off = self.src - self.tgt
            rot = (1 - e) * 2.4
            c, s = np.cos(rot), np.sin(rot)
            rx = off[:, 0] * c - off[:, 1] * s
            ry = off[:, 0] * s + off[:, 1] * c
            pos = self.tgt + np.stack([rx, ry], 1) * (1 - e)[:, None]
            alpha = np.clip(t / 0.4, 0, 1) * (0.55 + 0.45 * e)
        else:
            k = (t - T_PART) / 0.8
            pos = self.tgt + self.burst * ease_out(k) + (self.tgt - self.cen) * 0.4 * ease_out(k)
            alpha = np.full(len(self.tgt), max(0, 1 - k))
        for (x, y), r, col, a in zip(pos / 2, self.size, self.col, alpha):
            if a <= 0.01: continue
            cc = tuple(int(v * a) for v in col)
            d.ellipse((x - r, y - r, x + r, y + r), fill=cc)
        glow = lay.filter(ImageFilter.GaussianBlur(4))
        lay = ImageChops.add(lay, glow)
        lay = ImageChops.add(lay, glow)
        return lay.resize((W, H), Image.BILINEAR)


# ---------- audio helpers ----------
def load_envelope(path):
    with wave.open(path) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        a = np.frombuffer(w.readframes(n), np.int16).astype(float) / 32768
    if ch > 1: a = a.reshape(-1, ch).mean(1)
    hop = sr // FPS
    env = np.array([np.sqrt((a[i:i + hop] ** 2).mean()) for i in range(0, len(a) - hop, hop)])
    return env / (env.max() + 1e-9)


def synth_sfx(events, total, sr=44100):
    """仮SE: 粒子のキラキラ、回転の風切り音、スライドのシュッ"""
    out = np.zeros(int(total * sr) + sr)
    rng = np.random.default_rng(1)

    def whoosh(t0, dur, vol, lo=0.02):
        n = int(dur * sr)
        noise = rng.normal(0, 1, n)
        k = np.linspace(0, 1, n)
        smooth = np.convolve(noise, np.ones(12) / 12, "same")
        hi = noise - smooth
        mix = (smooth * (1 - k) + hi * k * 0.6)
        env = np.sin(np.pi * k) ** 2
        i = int(t0 * sr); out[i:i + n] += mix[: len(out) - i] * env[: len(out) - i] * vol

    def chime(t0, f, dur, vol):
        n = int(dur * sr); tt = np.arange(n) / sr
        s = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 2 * tt)) * np.exp(-tt * 5)
        i = int(t0 * sr); out[i:i + n] += s[: len(out) - i] * vol

    for kind, t in events:
        if kind == "sparkle":
            for j in range(14):
                chime(t + j * 0.13, rng.choice([1568, 1760, 2093, 2349, 2637]), 0.6, 0.05)
        elif kind == "spin":
            whoosh(t, 1.2, 0.35); chime(t + 1.2, 1046, 1.2, 0.12); chime(t + 1.25, 1568, 1.2, 0.08)
        elif kind == "slide":
            whoosh(t, 0.45, 0.28)
        elif kind == "end":
            for j, f in enumerate([1046, 1318, 1568, 2093]):
                chime(t + j * 0.09, f, 1.4, 0.09)
    out = np.clip(out, -1, 1)
    return (out * 32767).astype(np.int16), sr


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", help="ナレーション音声 (wav)")
    ap.add_argument("--bgm", help="BGM (wav/mp3)")
    ap.add_argument("--preview", type=float)
    ap.add_argument("--out", default=os.path.join(ROOT, "out/hayato_review.mp4"))
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, "script.json"), encoding="utf-8"))
    D = cfg["photo_seconds"]
    ev = cfg["events"]
    C = load_chara()
    bg = make_bg()
    hdr = header_layer(cfg)

    t_intro_talk = T_PART + T_SPIN
    t_move = t_intro_talk + T_TALK
    t_p0 = t_move + T_MOVE - 0.2
    t_outro = t_p0 + D * len(ev)
    wave_body, wave_hand, wave_pivot = make_wave_parts(C["finger_closed"])
    total = t_outro + T_OUTRO
    print(f"total length: {total:.1f}s")

    # キャラ配置（中央 / 左下）
    ch_w, ch_h = C["cross_smirk"].size
    big_h, small_h = 700, 440
    big = (W / 2, 880, big_h)          # (中心x, 足元y, 高さ)
    small = (250, 1070, small_h)
    pbox = (int(W / 2 - big_h * ch_w / ch_h / 2), 880 - big_h, int(big_h * ch_w / ch_h), big_h)
    parts = Particles(C["cross_smirk"], pbox)

    # 字幕・カード類を事前生成
    intro_b = bubble(cfg["intro_line"], (300, 905, 1620, 1050), tail=False)
    cards = [panel_card(os.path.join(ROOT, "assets/photos", e["photo"]), e["month"], e["name"]) for e in ev]
    badges = [badge(e["month"], e["name"]) for e in ev]
    bubbles = [bubble(e["line"], (470, 880, 1850, 1045)) for e in ev]
    out_b = bubble(cfg["outro_line"], (300, 905, 1620, 1050), tail=False)

    # セリフ区間 (開始, 長さ) — 音声がない場合は文字数から推定
    def speak_len(s): return min(len(s) * 0.115, D - 0.5)
    speech = [(t_intro_talk + 0.2, min(len(cfg["intro_line"]) * 0.11, T_TALK - 0.3))]
    for i, e in enumerate(ev):
        speech.append((t_p0 + i * D + 0.35, speak_len(e["line"])))
    speech.append((t_outro + O_TALK0 + 0.1, O_TALK_LEN - 0.2))
    env = load_envelope(args.voice) if args.voice else None

    def mouth_open(t, fi):
        if env is not None:
            i = int(t * FPS)
            return i < len(env) and env[i] > 0.12
        for s, l in speech:
            if s <= t < s + l:
                return ((fi // 3) * 2654435761 % 7) > 2  # 疑似ランダム口パク
        return False

    def blink(t):
        return (t % 3.1) < 0.12

    cache = {}

    def chara_img(name, h):
        k = (name, h)
        if k not in cache:
            im = C[name]; cache[k] = im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)
        return cache[k]

    def render(t, fi):
        fr = bg.copy()
        # 背景を漂う粒
        d = ImageDraw.Draw(fr)
        for j in range(40):
            x = (j * 197 + t * (12 + j % 5 * 6)) % W
            y = (j * 131 + math.sin(t * 0.6 + j) * 30) % H
            r = 1 + j % 3
            d.ellipse((x - r, y - r, x + r, y + r), fill=(120, 190, 255, 60 + j % 4 * 25))
        fr.alpha_composite(hdr)

        # --- 写真パネル ---
        if t >= t_p0 - 0.01:
            k = int((min(t, t_outro) - t_p0) // D) if t < t_outro else len(ev) - 1
            k = min(max(k, 0), len(ev) - 1)
            seg_t = t - (t_p0 + k * D)
            if t >= t_outro:
                ot = t - t_outro
                # 最後の写真が退場
                if ot < SLIDE:
                    e1 = ease_out(ot / SLIDE)
                    c = cards[k].copy(); c.putalpha(ImageChops.multiply(c.getchannel("A"), Image.new("L", c.size, int(255 * (1 - e1)))))
                    fr.alpha_composite(c, (int(PANEL[0] - 40 - 300 * e1), PANEL[1] - 40))
            else:
                # 前の写真の退場
                if k > 0 and seg_t < SLIDE:
                    e1 = ease_out(seg_t / SLIDE)
                    c = cards[k - 1].copy(); c.putalpha(ImageChops.multiply(c.getchannel("A"), Image.new("L", c.size, int(255 * (1 - e1)))))
                    fr.alpha_composite(c, (int(PANEL[0] - 40 - 300 * e1), PANEL[1] - 40))
                e = ease_out(seg_t / SLIDE)
                fr.alpha_composite(cards[k], (int(PANEL[0] - 40 + (1 - e) * 1400), PANEL[1] - 40))
                # 月バッジ（ポップ）
                bt = (seg_t - 0.3) / 0.35
                if bt > 0:
                    s = ease_back(bt)
                    b = badges[k]
                    bw, bh = max(1, int(b.width * s)), max(1, int(b.height * s))
                    fr.alpha_composite(b.resize((bw, bh)), (int(PANEL[0] - 50 + (b.width - bw) / 2), int(PANEL[1] - 70 + (b.height - bh) / 2)))

        # --- 粒子 ---
        if t < T_PART + 0.8:
            fr = Image.fromarray(np.array(ImageChops.add(fr.convert("RGB"), parts.draw(t)))).convert("RGBA")
        ot = t - t_outro
        if O_DIS0 <= ot < O_DIS0 + O_DIS:
            pt = T_PART * (1 - (ot - O_DIS0) / O_DIS)  # 集合の逆再生 = 粒子になって散る
            fr = Image.fromarray(np.array(ImageChops.add(fr.convert("RGB"), parts.draw(pt)))).convert("RGBA")

        # --- キャラ ---
        mo = mouth_open(t, fi)
        if t < T_PART:
            pass
        elif t < t_intro_talk:
            st = (t - T_PART) / T_SPIN
            e = ease_out(st)
            # 体の真ん中を軸にくるくる回りながら大きくなって登場
            ang = (1 - e) * 720
            s = 0.15 + 0.85 * ease_back(st)
            h = max(4, int(big_h * s))
            im = chara_img("cross_smirk", big_h).resize((max(2, int(big_h * ch_w / ch_h * s)), h), Image.BILINEAR)
            im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
            ccx, ccy = big[0], big[1] - big_h / 2   # 回転の中心（キャラの中心）
            pos = (int(ccx - im.width / 2), int(ccy - im.height / 2))
            # 回転中の青い光
            glow = Image.new("RGBA", im.size, CYAN + (0,))
            glow.putalpha(im.getchannel("A").point(lambda v: int(v * (1 - e) * 0.9)))
            fr.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)), pos)
            fr.alpha_composite(im, pos)
        elif ot >= O_WAVE0:
            if ot < O_SPIN0:
                # バイバイ（手首を支点に手を振る）
                ang = 15 * math.sin((ot - O_WAVE0) * 2 * math.pi * 2.4) * min(1, (ot - O_WAVE0) / 0.15)
                base = wave_body.copy()
                hand = wave_hand.rotate(ang, resample=Image.BICUBIC, center=wave_pivot)
                base.alpha_composite(hand)
                h = big_h
                im = base.resize((int(base.width * h / base.height), h), Image.LANCZOS)
                bob = math.sin(t * 2.2) * 4
                fr.alpha_composite(im, (int(big[0] - im.width / 2), int(big[1] - h + bob)))
            elif ot < O_SPIN0 + O_SPIN:
                st = (ot - O_SPIN0) / O_SPIN
                e = st * st
                theta = e * 5 * math.pi
                sx = math.cos(theta)
                h = big_h
                im = chara_img("finger_closed", h)
                if sx < 0: im = im.transpose(Image.FLIP_LEFT_RIGHT)
                w = max(2, int(im.width * abs(sx)))
                im = im.resize((w, h))
                fade_a = 1 - ease_in_out((st - 0.35) / 0.65)
                glow = Image.new("RGBA", im.size, CYAN + (0,))
                glow.putalpha(im.getchannel("A").point(lambda v: int(v * min(1, st * 1.5) * 0.95)))
                fr.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)), (int(big[0] - w / 2), int(big[1] - h)))
                if fade_a > 0:
                    im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * fade_a)))
                    fr.alpha_composite(im, (int(big[0] - w / 2), int(big[1] - h)))
        else:
            if ot >= O_MOVE0:
                e = ease_in_out((ot - O_MOVE0) / (O_MOVE1 - O_MOVE0))
                cx = small[0] + (big[0] - small[0]) * e
                fy = small[1] + (big[1] - small[1]) * e
                h = int(small[2] + (big[2] - small[2]) * e); pose = "cross"
            elif t < t_move:
                cx, fy, h = big; pose = "cross"
            elif t < t_move + T_MOVE:
                e = ease_in_out((t - t_move) / T_MOVE)
                cx = big[0] + (small[0] - big[0]) * e
                fy = big[1] + (small[1] - big[1]) * e
                h = int(big[2] + (small[2] - big[2]) * e); pose = "cross"
            else:
                cx, fy, h = small; pose = "right"
            if mo:
                name = f"{pose}_open"
            elif blink(t):
                name = f"{pose}_closed"
            else:
                name = f"{pose}_smirk"
            bob = math.sin(t * 2.2) * 4  # ゆらゆら
            im = chara_img(name, int(h))
            fr.alpha_composite(im, (int(cx - im.width / 2), int(fy - im.height + bob)))

        # --- 吹き出し / 字幕 ---
        def put_bubble(bb, a):
            img, pos = bb
            if a < 1:
                img = img.copy(); img.putalpha(ImageChops.multiply(img.getchannel("A"), Image.new("L", img.size, int(255 * a))))
            fr.alpha_composite(img, pos)
        if t_intro_talk + 0.1 <= t < t_move:
            put_bubble(intro_b, min(1, (t - t_intro_talk - 0.1) / 0.25) * min(1, (t_move - t) / 0.2))
        elif t_p0 <= t < t_outro:
            k = min(int((t - t_p0) // D), len(ev) - 1)
            st = t - (t_p0 + k * D)
            put_bubble(bubbles[k], min(1, max(0, st - 0.25) / 0.25) * min(1, (D - st) / 0.15))
        elif t_outro + O_TALK0 <= t < t_outro + O_WAVE0 + 0.5:
            put_bubble(out_b, min(1, (ot - O_TALK0) / 0.25) * min(1, (O_WAVE0 + 0.5 - ot) / 0.25))

        # 冒頭・最後のフェード
        fade = min(1, t / 0.3) * min(1, (total - t) / T_FADE)
        rgb = fr.convert("RGB")
        if fade < 1:
            rgb = Image.eval(rgb, lambda v: int(v * fade))
        return rgb

    if args.preview is not None:
        p = os.path.join(ROOT, "out/preview.png")
        render(args.preview, int(args.preview * FPS)).save(p); print(p); return

    # SE
    sfx_ev = [("sparkle", 0.2), ("spin", T_PART - 0.1), ("slide", t_move)]
    sfx_ev += [("slide", t_p0 + i * D) for i in range(len(ev))] + [("slide", t_outro), ("spin", t_outro + O_SPIN0 - 0.1), ("sparkle", t_outro + O_DIS0)]
    pcm, sr = synth_sfx(sfx_ev, total)
    sfx_path = os.path.join(ROOT, "out/sfx.wav")
    with wave.open(sfx_path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

    tmp = os.path.join(ROOT, "out/_video.mp4")
    nfr = int(total * FPS)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for fi in range(nfr):
        p.stdin.write(render(fi / FPS, fi).tobytes())
        if fi % 90 == 0: print(f"  frame {fi}/{nfr}", flush=True)
    p.stdin.close(); p.wait()

    inputs = ["-i", tmp, "-i", sfx_path]
    filt = "[1:a]volume=1.0[s]"
    mix = ["[s]"]
    n = 2
    if args.voice:
        inputs += ["-i", args.voice]; filt += f";[{n}:a]volume=1.0[v]"; mix.append("[v]"); n += 1
    if args.bgm:
        inputs += ["-i", args.bgm]; filt += f";[{n}:a]volume=0.25,afade=t=out:st={total-1.5}:d=1.5[b]"; mix.append("[b]"); n += 1
    filt += f";{''.join(mix)}amix=inputs={len(mix)}:normalize=0,atrim=0:{total}[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", args.out], check=True)
    os.remove(tmp)
    print("done:", args.out)


if __name__ == "__main__":
    main()
