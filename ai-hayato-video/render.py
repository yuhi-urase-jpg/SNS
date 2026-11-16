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
T_SPIN = 1.1      # 回転登場
T_TALK = 2.6      # 中央で挨拶
T_MOVE = 0.8      # 左下へ移動
SLIDE = 0.55      # 写真スライドイン
# エンディング（t_outro からの相対秒）
O_MOVE0, O_MOVE1 = 0.25, 1.05   # 中央へ戻る
# 締めのセリフ3つ (開始, 長さ)：①お辞儀 ②手を振りながら ③「あとは任せたよ！」
O_LINES = [(1.15, 1.8), (3.05, 1.8), (4.95, 2.8)]
O_TALK0 = O_LINES[0][0]
O_WAVE0 = O_LINES[-1][0] + O_LINES[-1][1] + 0.05   # バイバイ
O_WAVE = 0.5
O_SPIN0 = O_WAVE0 + O_WAVE      # 回転して粒子に
O_SPIN = 1.1
O_DIS0 = O_SPIN0 + 0.55         # 粒子化スタート
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
CROP = (470, 15, 1170, 935)  # 全ポーズ共通の切り抜き枠（ポーズを切り替えても位置がずれない）


def load_chara():
    d = os.path.join(ROOT, "assets/chara")
    out = {}
    for f in os.listdir(d):
        im = Image.open(os.path.join(d, f)).convert("RGBA").crop(CROP)
        a = np.array(im)
        a[..., 3] = np.where(a[..., 3] < 40, 0, a[..., 3])  # 薄いゴミを除去
        out[f[:-4]] = Image.fromarray(a)
    return out


def make_wave_parts(C):
    """バイバイ画像から「手（袖口まで）」を切り出し、手首を支点に振れるようにする"""
    ox, oy = CROP[0], CROP[1]
    a = np.array(C["wave_smirk"])
    alpha = a[..., 3] > 100
    box = np.zeros_like(alpha); box[420 - oy:552 - oy, 500 - ox:652 - ox] = True
    region = alpha & box
    m = np.zeros_like(alpha); m[480 - oy, 575 - ox] = True
    for _ in range(600):
        n = m.copy(); n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        n &= region
        if (n == m).all(): break
        m = n
    for _ in range(3):
        n = m.copy(); n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n & box
    hand = a.copy(); hand[..., 3] = np.where(m, a[..., 3], 0)
    bodies = {}
    for k in ("wave_smirk", "wave_closed", "wave_open", "wave_laugh"):
        b = np.array(C[k]); b[..., 3] = np.where(m, 0, b[..., 3]); bodies[k] = Image.fromarray(b)
    return bodies, Image.fromarray(hand), (602 - ox, 550 - oy)


class Background:
    """青い粒子の星雲 + 浮遊する結晶片・泡 + 横ライン（AI・未来感）"""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # ベース（ほぼ黒の紺）
        y = np.linspace(-1, 1, H)[:, None]; x = np.linspace(-1, 1, W)[None, :]
        r = np.sqrt((x * 0.9) ** 2 + y ** 2)
        base = np.stack([2 + 6 * (1 - r), 6 + 16 * (1 - r), 16 + 40 * (1 - r)], -1).clip(0, 255)
        self.base = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(g)
        # 細い横ライン（ゆるくカーブ）
        for gy in range(-20, H + 40, 34):
            pts = [(xx, gy + 18 * math.sin(xx / W * math.pi) * (1 if gy < H / 2 else -1) * abs(gy - H / 2) / (H / 2)) for xx in range(0, W + 1, 60)]
            d.line(pts, fill=(170, 200, 255, 38), width=1)
        # うっすら同心円
        for rr, a in [(330, 40), (420, 26), (520, 18)]:
            d.ellipse((W / 2 - rr, H / 2 - rr, W / 2 + rr, H / 2 + rr), outline=(150, 200, 255, a), width=2)
        self.base.alpha_composite(g)
        self.nebula = self._make_nebula(rng)
        # 星雲の中の細かいチリ（フル解像度）
        n = 14000
        dx = rng.normal(0, 0.42, n) * W / 2 * 0.66; dy = rng.normal(0, 0.42, n) * H / 2 * 0.98
        px = np.clip(W / 2 + dx, 0, W - 1).astype(int); py = np.clip(H / 2 + dy, 0, H - 1).astype(int)
        dust = np.zeros((H, W, 4), np.uint8)
        v = rng.uniform(0, 1, n) ** 2
        dust[py, px, 0] = (150 + 100 * v).astype(np.uint8); dust[py, px, 1] = (195 + 60 * v).astype(np.uint8)
        dust[py, px, 2] = 255; dust[py, px, 3] = (40 + 200 * v).astype(np.uint8)
        self.dust = Image.fromarray(dust)
        # 結晶片
        n = 70
        self.frag = dict(
            ang=rng.uniform(0, 2 * np.pi, n), rad=rng.uniform(80, 1100, n), size=rng.choice([4, 6, 9, 13, 18, 26], n, p=[.25, .25, .2, .15, .1, .05]),
            spd=rng.uniform(6, 22, n), rot=rng.uniform(0, 360, n), rspd=rng.uniform(-60, 60, n), kind=rng.integers(0, 3, n),
            bright=rng.uniform(0.55, 1, n))
        m = 30
        self.bub = dict(x=rng.uniform(0, W, m), y=rng.uniform(0, H, m), r=rng.uniform(8, 22, m), spd=rng.uniform(8, 25, m), ph=rng.uniform(0, 6, m))

    def _make_nebula(self, rng):
        w, h = int(W // 2 * 1.5), int(H // 2 * 1.5)   # 回転しても端が欠けないよう大きめに作る
        # フラクタルノイズ
        acc = np.zeros((h, w))
        for o, amp in [(4, 1.0), (8, 0.55), (16, 0.35), (32, 0.22), (64, 0.14), (128, 0.08)]:
            g = rng.random((o * h // w + 2, o + 2))
            im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
            acc += np.array(im) / 255 * amp
        acc = (acc - acc.min()) / (acc.max() - acc.min())
        y = np.linspace(-1.5, 1.5, h)[:, None]; x = np.linspace(-1.5, 1.5, w)[None, :]
        th = np.arctan2(y, x)
        wob = 1 + 0.12 * np.sin(th * 3 + 1) + 0.08 * np.sin(th * 5 + 2) + (acc - 0.5) * 0.35
        r = np.sqrt((x / 0.66) ** 2 + (y / 0.98) ** 2) / wob
        shape = np.clip((1.0 - r) / 0.25, 0, 1) ** 1.2           # 不規則な輪郭の大きな雲
        rim = np.exp(-((r - 0.78) / 0.2) ** 2)                     # 外周がやや明るい
        v = np.clip(shape * (0.25 + 0.75 * acc ** 1.3) + rim * acc * 0.6, 0, 1)
        col = np.stack([20 + 120 * v ** 2.4, 60 + 130 * v ** 1.5, 170 + 80 * v ** 0.6], -1).clip(0, 255)
        alpha = (np.clip(v * 1.3, 0, 1) * 140).astype(np.uint8)
        neb = Image.fromarray(np.dstack([col.astype(np.uint8), alpha]), "RGBA")
        neb = neb.filter(ImageFilter.GaussianBlur(0.6))
        return neb

    def _poly(self, kind, cx, cy, s, rot):
        k = 6 if kind == 0 else 4
        return [(cx + s * math.cos(math.radians(rot + i * 360 / k)), cy + s * math.sin(math.radians(rot + i * 360 / k)) * (0.8 if kind == 2 else 1)) for i in range(k)]

    def frame(self, t):
        fr = self.base.copy()
        # 星雲：ゆっくり回転 + 呼吸
        sc = 1.0 + 0.03 * math.sin(t * 0.8)
        nw, nh = self.nebula.size
        neb = self.nebula.rotate(t * 2.5, resample=Image.BILINEAR)
        cw, chh = W / 2 / sc, H / 2 / sc
        neb = neb.crop((int(nw / 2 - cw / 2), int(nh / 2 - chh / 2), int(nw / 2 + cw / 2), int(nh / 2 + chh / 2))).resize((W, H), Image.BILINEAR)
        fr.alpha_composite(neb)
        fr.alpha_composite(self.dust)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))   # 半透明を正しく重ねるため別レイヤーに描く
        d = ImageDraw.Draw(ov)
        F = self.frag
        for i in range(len(F["ang"])):
            rr = (F["rad"][i] + F["spd"][i] * t) % 1150 + 60
            cx = W / 2 + math.cos(F["ang"][i]) * rr * 1.2
            cy = H / 2 + math.sin(F["ang"][i]) * rr * 0.75
            s = F["size"][i]; rot = F["rot"][i] + F["rspd"][i] * t
            b = F["bright"][i]
            pts = self._poly(F["kind"][i], cx, cy, s, rot)
            if F["kind"][i] == 1:
                d.polygon(pts, fill=(int(20 * b), int(40 * b), int(80 * b), 220), outline=(90, 140, 220, 160))
            else:
                d.polygon(pts, fill=(int(235 * b), int(240 * b), int(250 * b), 235))
                d.polygon(pts[: len(pts) // 2 + 1] + [(cx, cy)], fill=(int(190 * b), int(205 * b), int(230 * b), 235))
        B = self.bub
        for i in range(len(B["x"])):
            x = (B["x"][i] + math.sin(t * 0.7 + B["ph"][i]) * 20) % W
            y = (B["y"][i] - B["spd"][i] * t) % H
            r = B["r"][i]
            d.ellipse((x - r, y - r, x + r, y + r), fill=(120, 170, 230, 10), outline=(200, 225, 255, 110), width=1)
            d.ellipse((x - r * 0.45 - 2, y - r * 0.45 - 2, x - r * 0.45 + 2, y - r * 0.45 + 2), fill=(255, 255, 255, 170))
        fr.alpha_composite(ov)
        return fr


def header_layer(cfg):
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    text_bold(d, (1850, 34), cfg["header"], 24, CYAN + (255,), anchor="rt")
    text_bold(d, (1850, 68), cfg["title"], 44, (255, 255, 255, 255), anchor="rt")
    d.line([(1850 - 560, 128), (1850, 128)], fill=CYAN + (120,), width=2)
    return L


PANEL = (560, 165, 1850, 855)


def panel_card(photo_path, month, name, focus=(0.5, 0.5)):
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
    pad = 10
    iw, ih = pw - pad * 2, ph - pad * 2
    if os.path.exists(photo_path):
        ph_img = Image.open(photo_path).convert("RGB")
    else:  # 写真未着の仮枠
        ph_img = Image.new("RGB", (1600, 900), (30, 50, 90))
        dd = ImageDraw.Draw(ph_img)
        text_bold(dd, (800, 400), f"{month} {name}", 90, (200, 230, 255), anchor="mm")
        text_bold(dd, (800, 530), "（写真差し替え予定）", 56, (140, 180, 230), anchor="mm")
    # 枠いっぱいに拡大し（cover）、人が写っている位置(focus)を中心に切り抜く
    fx, fy = focus
    sc = max(iw / ph_img.width, ih / ph_img.height)
    big_img = ph_img.resize((max(iw, round(ph_img.width * sc)), max(ih, round(ph_img.height * sc))), Image.LANCZOS)
    cx0 = int(min(max(fx * big_img.width - iw / 2, 0), big_img.width - iw))
    cy0 = int(min(max(fy * big_img.height - ih / 2, 0), big_img.height - ih))
    blur = big_img.crop((cx0, cy0, cx0 + iw, cy0 + ih))
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
    """細かい光の粒（サブピクセル描画 + 発光 + きらめき）"""

    def __init__(self, chara_img, box, n=9000, seed=3):
        rng = np.random.default_rng(seed)
        x0, y0, w, h = box
        m = np.array(chara_img.resize((w, h)).getchannel("A")) > 128
        ys, xs = np.nonzero(m)
        idx = rng.choice(len(xs), n)
        self.tgt = np.stack([xs[idx] + x0 + rng.uniform(-.5, .5, n), ys[idx] + y0 + rng.uniform(-.5, .5, n)], 1)
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = rng.uniform(450, 1400, n)
        cen = np.array([W / 2, H / 2])
        self.src = cen + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        self.delay = rng.uniform(0, 0.45, n)
        self.bri = np.clip(rng.lognormal(-0.2, 0.55, n), 0.2, 4.0)
        pal = np.array([[0.62, 0.86, 1.0], [0.30, 0.62, 1.0], [0.14, 0.42, 1.0], [0.88, 0.96, 1.0]])
        self.col = pal[rng.choice(4, n, p=[.35, .3, .15, .2])]
        self.tw = rng.uniform(6, 18, n); self.ph = rng.uniform(0, 6.3, n)
        self.burst = rng.normal(0, 1, (n, 2)) * 160
        self.cen = np.array([x0 + w / 2, y0 + h / 2])

    def draw(self, t):
        if t <= T_PART:
            p = np.clip((t - self.delay) / (T_PART - 0.45), 0, 1)
            e = 3 * p * p - 2 * p ** 3
            off = self.src - self.tgt
            rot = (1 - e) * 2.4
            c, s = np.cos(rot), np.sin(rot)
            rx = off[:, 0] * c - off[:, 1] * s
            ry = off[:, 0] * s + off[:, 1] * c
            pos = self.tgt + np.stack([rx, ry], 1) * (1 - e)[:, None]
            alpha = np.clip(t / 0.4, 0, 1) * (0.5 + 0.5 * e)
        else:
            k = (t - T_PART) / 0.8
            pos = self.tgt + self.burst * ease_out(k) + (self.tgt - self.cen) * 0.4 * ease_out(k)
            alpha = np.full(len(self.tgt), max(0, 1 - k))
        b = alpha * self.bri * (0.6 + 0.4 * np.sin(t * self.tw + self.ph))
        x, y = pos[:, 0], pos[:, 1]
        ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1) & (b > 0.01)
        x, y, b, col = x[ok], y[ok], b[ok], self.col[ok]
        xi, yi = x.astype(int), y.astype(int); fx, fy = x - xi, y - yi
        idx = np.concatenate([yi * W + xi, yi * W + xi + 1, (yi + 1) * W + xi, (yi + 1) * W + xi + 1])
        wgt = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * np.tile(b, 4)
        acc = np.stack([np.bincount(idx, wgt * np.tile(col[:, c], 4), minlength=W * H) for c in range(3)], -1).reshape(H, W, 3)
        small = acc.reshape(H // 4, 4, W // 4, 4, 3).sum((1, 3))
        glow = Image.fromarray(np.clip(small * 120, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2)).resize((W, H), Image.BILINEAR)
        core = np.clip(acc * 430, 0, 255).astype(np.uint8)
        return ImageChops.add(Image.fromarray(core), glow)


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
    bg = Background()
    hdr = header_layer(cfg)

    t_intro_talk = T_PART + T_SPIN
    t_move = t_intro_talk + T_TALK
    t_p0 = t_move + T_MOVE - 0.2
    t_outro = t_p0 + D * len(ev)
    wave_bodies, wave_hand, wave_pivot = make_wave_parts(C)
    total = t_outro + T_OUTRO
    print(f"total length: {total:.1f}s")

    # キャラ配置（中央 / 左下）
    ch_w, ch_h = C["cross_smirk"].size
    big_h, small_h = 700, 440
    big = (W / 2, 880, big_h)          # (中心x, 足元y, 高さ)
    small = (250, 1070, small_h)
    pbox = (int(W / 2 - big_h * ch_w / ch_h / 2), 880 - big_h, int(big_h * ch_w / ch_h), big_h)
    parts = Particles(C["cross_smirk"], pbox)
    sc2 = 0.5
    pbox2 = (int(W / 2 - big_h * sc2 * ch_w / ch_h / 2), int(880 - big_h / 2 - big_h * sc2 / 2), int(big_h * sc2 * ch_w / ch_h), int(big_h * sc2))
    parts_end = Particles(C["wave_smirk"], pbox2, seed=11)

    # 字幕・カード類を事前生成
    intro_b = bubble(cfg["intro_line"], (300, 905, 1620, 1050), tail=False)
    cards = [panel_card(os.path.join(ROOT, "assets/photos", e["photo"]), e["month"], e["name"], tuple(e.get("focus", (0.5, 0.5)))) for e in ev]
    badges = [badge(e["month"], e["name"]) for e in ev]
    bubbles = [bubble(e["line"], (470, 880, 1850, 1045)) for e in ev]
    out_bs = [bubble(l, (300, 905, 1620, 1050), tail=False) for l in cfg["outro_lines"]]

    # セリフ区間 (開始, 長さ) — 音声がない場合は文字数から推定
    def speak_len(s): return min(len(s) * 0.115, D - 0.5)
    speech = [(t_intro_talk + 0.2, min(len(cfg["intro_line"]) * 0.11, T_TALK - 0.3))]
    for i, e in enumerate(ev):
        speech.append((t_p0 + i * D + 0.35, speak_len(e["line"])))
    for (s0, ln) in O_LINES[1:]:
        speech.append((t_outro + s0 + 0.1, ln - 0.3))
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
        fr = bg.frame(t)
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
            fr = Image.fromarray(np.array(ImageChops.add(fr.convert("RGB"), parts_end.draw(pt)))).convert("RGBA")

        # --- キャラ ---
        mo = mouth_open(t, fi)
        if t < T_PART:
            pass
        elif t < t_intro_talk:
            st = (t - T_PART) / T_SPIN
            e = ease_out(st)
            # 体の真ん中を軸に高速回転しながら大きくなって登場（5回転、最後にキュッと止まる）
            ang = (1 - (1 - (1 - st) ** 2.2)) * 1800
            s = 0.12 + 0.88 * ease_back(st)
            h = max(4, int(big_h * s))
            base_im = chara_img("cross_smirk", big_h).resize((max(2, int(big_h * ch_w / ch_h * s)), h), Image.BILINEAR)
            ccx, ccy = big[0], big[1] - big_h / 2   # 回転の中心（キャラの中心）
            speed = 1 - st                          # 速いほど残像を強く
            for lag, a in [(28, 0.22), (14, 0.4)]:   # 残像（モーションブラー風）
                if speed < 0.15: break
                g = base_im.rotate(ang + lag * speed * 1.8, resample=Image.BILINEAR, expand=True)
                g.putalpha(g.getchannel("A").point(lambda v, a=a: int(v * a * speed)))
                fr.alpha_composite(g, (int(ccx - g.width / 2), int(ccy - g.height / 2)))
            im = base_im.rotate(ang, resample=Image.BICUBIC, expand=True)
            pos = (int(ccx - im.width / 2), int(ccy - im.height / 2))
            glow = Image.new("RGBA", im.size, CYAN + (0,))
            glow.putalpha(im.getchannel("A").point(lambda v: int(v * speed * 0.9)))
            fr.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)), pos)
            fr.alpha_composite(im, pos)
        elif ot >= O_MOVE1:
            ccx, ccy = big[0], big[1] - big_h / 2
            if ot < O_LINES[1][0] - 0.05:
                # 「今年も1年間お疲れ様でした」でお辞儀
                bt = ot - O_MOVE1
                im = chara_img("bow", big_h)
                k = ease_back(min(1, bt / 0.3))
                hh = int(big_h * (0.93 + 0.07 * k))
                im = im.resize((int(im.width * hh / big_h), hh))
                fr.alpha_composite(im, (int(big[0] - im.width / 2), int(big[1] - hh)))
            elif ot < O_SPIN0:
                # 中央でセリフ → バイバイ（手首を支点に手を振る）
                bye = ot >= O_WAVE0
                amp = 16 if bye else 8
                ang = amp * math.sin((ot - O_MOVE1) * 2 * math.pi * (2.6 if bye else 1.6))
                if bye:
                    # 目を開けたまま「バイバイ」、ときどき瞬き
                    bt = ot - O_WAVE0
                    if 0.55 <= bt % 1.1 < 0.65:
                        face = "wave_closed"
                    else:
                        face = "wave_open" if int(bt / 0.3) % 2 == 0 else "wave_smirk"
                elif mo:
                    face = "wave_open"
                elif blink(t):
                    face = "wave_closed"
                else:
                    face = "wave_smirk"
                base = wave_bodies[face].copy()
                base.alpha_composite(wave_hand.rotate(ang, resample=Image.BICUBIC, center=wave_pivot))
                h = big_h
                im = base.resize((int(base.width * h / base.height), h), Image.LANCZOS)
                pop = ease_back(min(1, (ot - O_LINES[1][0] + 0.05) / 0.25))   # ポーズ切替時に軽く弾む
                if pop < 1:
                    hh = int(h * (0.94 + 0.06 * pop)); im = im.resize((int(im.width * hh / h), hh)); h = hh
                bob = math.sin(t * 2.2) * 4
                fr.alpha_composite(im, (int(big[0] - im.width / 2), int(big[1] - h + bob)))
            elif ot < O_SPIN0 + O_SPIN:
                # 登場と同じ「真ん中軸の高速回転」で小さくなりながら消える
                st = (ot - O_SPIN0) / O_SPIN
                ang = st ** 2.2 * 1800
                s = max(0.08, 1 - 0.9 * st ** 1.6)
                h = max(4, int(big_h * s))
                src = chara_img("wave_smirk", big_h)
                base_im = src.resize((max(2, int(src.width * s)), h), Image.BILINEAR)
                speed = st
                fade_a = 1 - ease_in_out((st - 0.55) / 0.45)
                for lag, a_ in [(28, 0.22), (14, 0.4)]:
                    if speed < 0.15: break
                    g = base_im.rotate(ang - lag * speed * 1.8, resample=Image.BILINEAR, expand=True)
                    g.putalpha(g.getchannel("A").point(lambda v, a_=a_: int(v * a_ * speed * fade_a)))
                    fr.alpha_composite(g, (int(ccx - g.width / 2), int(ccy - g.height / 2)))
                im = base_im.rotate(ang, resample=Image.BICUBIC, expand=True)
                pos = (int(ccx - im.width / 2), int(ccy - im.height / 2))
                glow = Image.new("RGBA", im.size, CYAN + (0,))
                glow.putalpha(im.getchannel("A").point(lambda v: int(v * min(1, st * 1.4) * 0.95)))
                fr.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)), pos)
                if fade_a > 0:
                    im.putalpha(im.getchannel("A").point(lambda v: int(v * fade_a)))
                    fr.alpha_composite(im, pos)
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
        elif t_outro + O_TALK0 <= t < t_outro + O_WAVE0:
            for (s0, ln), bb in zip(O_LINES, out_bs):
                if s0 <= ot < s0 + ln + 0.1:
                    put_bubble(bb, min(1, (ot - s0) / 0.2) * min(1, (s0 + ln + 0.1 - ot) / 0.15))

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
