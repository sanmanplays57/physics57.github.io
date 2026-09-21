"""The strip, conformally mapped onto the upper half plane and onto the disk.

Take the strip S = {w = u + iv : 0 < v < pi}. Then z = exp(w) carries it onto
the upper half plane, since exp(u + iv) has modulus e^u and argument v, and
the Cayley map (z - i)/(z + i) carries the upper half plane onto the unit
disk. Composing, the strip goes onto the disk by (e^w - i)/(e^w + i).

What a uniform grid on the strip becomes is the whole content of the picture:

  - a horizontal line v = const goes to the ray of argument v, and then to a
    circular arc in the disk;
  - a vertical segment u = const goes to the semicircle of radius e^u, and
    then to an arc cutting the first family at right angles;
  - the two edges v = 0 and v = pi go to the positive and negative real axis,
    and so to the lower and upper halves of the unit circle;
  - the midline v = pi/2 goes to the positive imaginary axis, and so to the
    horizontal diameter of the disk.

The small squares are there to show what conformal means pointwise: they are
carried to squares again, only rotated and rescaled by the local factor
|f'(w)|, which is why the ones further right come out larger.

Intermediate frames of each morph interpolate the image points linearly, so
only the endpoints of a morph are the actual conformal maps.

Render (fast, low res):
    python -m manim -ql manim/conformal_strip.py ConformalStrip
Render (1080p60):
    python -m manim -qh manim/conformal_strip.py ConformalStrip
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

INK = "#E8EEF7"
MUTED = "#97A3B6"
TEAL = "#2DD4BF"            # the horizontal family
WBLUE = "#5B9BFF"           # the vertical family
AMBER = "#F5A524"           # the edge v = 0
PINK = "#F472B6"            # the edge v = pi
GOLD = "#FDE047"            # the little squares

U0, U1 = -2.2, 2.2          # how far along the strip the grid runs
NU, NV = 12, 8              # gaps, so NU + 1 verticals and NV + 1 horizontals
SAMP = 120                  # samples per grid line

SW, STRIP_Y = 0.95, 0.10    # placement of the strip
US, UHP_Y = 0.33, -1.70     # of the upper half plane, whose origin sits low
DR, DISK_Y = 2.30, 0.0      # and of the disk


def strip_pt(w):
    return np.array([w.real * SW, (w.imag - PI / 2) * SW + STRIP_Y, 0.0])


def uhp_pt(w):
    z = np.exp(w)
    return np.array([z.real * US, z.imag * US + UHP_Y, 0.0])


def disk_pt(w):
    z = np.exp(w)
    c = (z - 1j) / (z + 1j)
    return np.array([c.real * DR, c.imag * DR + DISK_Y, 0.0])


def horizontals():
    out = []
    for j in range(NV + 1):
        v = PI * j / NV
        ws = [complex(U0 + (U1 - U0) * i / (SAMP - 1), v) for i in range(SAMP)]
        if j == 0:
            out.append((ws, AMBER, 6.0))
        elif j == NV:
            out.append((ws, PINK, 6.0))
        else:
            out.append((ws, TEAL, 2.2))
    return out


def verticals():
    out = []
    for k in range(NU + 1):
        u = U0 + (U1 - U0) * k / NU
        ws = [complex(u, PI * i / (SAMP - 1)) for i in range(SAMP)]
        out.append((ws, WBLUE, 2.2))
    return out


LINES = horizontals() + verticals()

# Placed to the right, where the local scale factor e^u is big enough to see.
# They have to stay small: conformality is a statement about the limit, and a
# square wide enough that e^u varies across it comes out a visible trapezoid.
SQUARES = [(1.15, PI / 4), (1.15, 3 * PI / 4), (1.9, PI / 2)]
SIDE = 0.16


def square_ws(u, v, s=SIDE):
    """A small square in the w plane, sampled along its four sides."""
    h = 0.5 * s
    corner = [(u - h, v - h), (u + h, v - h), (u + h, v + h),
              (u - h, v + h), (u - h, v - h)]
    ws = []
    for (a0, b0), (a1, b1) in zip(corner[:-1], corner[1:]):
        for i in range(14):
            f = i / 14
            ws.append(complex(a0 + f * (a1 - a0), b0 + f * (b1 - b0)))
    ws.append(complex(*corner[-1]))
    return ws


SQ_WS = [square_ws(u, v) for u, v in SQUARES]


def disk_rim():
    """The full unit circle, in the colours of the two edges it comes from.

    The grid only runs over a finite range of u, so its edges stop short of
    the points +1 and -1, which are the images of u = +inf and u = -inf. The
    rim closes that gap without pretending the grid reaches it.
    """
    centre = np.array([0.0, DISK_Y, 0.0])
    return VGroup(
        Arc(radius=DR, start_angle=PI, angle=PI, arc_center=centre,
            stroke_color=AMBER, stroke_width=6.0),
        Arc(radius=DR, start_angle=0.0, angle=PI, arc_center=centre,
            stroke_color=PINK, stroke_width=6.0))


def picture(a, f, g, squares=True):
    """The grid, with each point a fraction `a` of the way from f to g."""
    grp = VGroup()
    for ws, col, wd in LINES:
        m = VMobject(stroke_color=col, stroke_width=wd, fill_opacity=0.0)
        m.set_points_as_corners(
            [(1.0 - a) * f(w) + a * g(w) for w in ws])
        grp.add(m)
    if squares:
        for ws in SQ_WS:
            m = VMobject(stroke_color=GOLD, stroke_width=3.5, fill_opacity=0.0)
            m.set_points_as_corners(
                [(1.0 - a) * f(w) + a * g(w) for w in ws])
            grp.add(m)
    return grp


class ConformalStrip(Scene):
    def construct(self):
        title = Tex("Conformal maps of the strip", color=INK).scale(0.95)
        title.to_edge(UP, buff=0.42)
        self.play(FadeIn(title, shift=0.3 * DOWN), run_time=1.0)

        # ------------------------------------------------ 1. the strip itself
        strip_lab = MathTex(r"S=\{\,w=u+iv\ :\ 0<v<\pi\,\}",
                            color=MUTED).scale(0.68)
        strip_lab.to_edge(DOWN, buff=0.55)

        static = picture(0.0, strip_pt, strip_pt)
        self.play(LaggedStart(*[Create(m) for m in static], lag_ratio=0.03),
                  run_time=2.6)
        self.play(Write(strip_lab), run_time=1.0)
        self.wait(0.9)

        edge_note = Tex(r"the edges $v=0$ and $v=\pi$ are marked",
                        color=MUTED).scale(0.58)
        edge_note.next_to(strip_lab, UP, buff=0.28)
        self.play(FadeIn(edge_note), run_time=0.7)
        self.wait(1.2)
        self.play(FadeOut(edge_note), run_time=0.5)

        # ------------------------------------- 2. exponentiate onto the plane
        exp_lab = MathTex(r"z=e^{\,w}", color=TEAL).scale(0.95)
        exp_lab.to_corner(UL, buff=0.75).shift(0.55 * DOWN)

        axis = Line([-6.9, UHP_Y, 0.0], [6.9, UHP_Y, 0.0],
                    stroke_color=MUTED, stroke_width=1.6, stroke_opacity=0.45)

        self.play(FadeOut(strip_lab), FadeIn(exp_lab), run_time=0.7)

        t = ValueTracker(0.0)
        live = always_redraw(lambda: picture(t.get_value(), strip_pt, uhp_pt))
        # Create added the grid lines one by one, so the group itself was
        # never in the scene and removing it would leave every line behind.
        self.remove(*static)
        self.add(live)
        self.play(t.animate.set_value(1.0), FadeIn(axis),
                  run_time=3.6, rate_func=smooth)
        self.wait(0.6)

        uhp_note = VGroup(
            MathTex(r"\operatorname{Im}z>0", color=INK).scale(0.66),
            Tex(r"$v=$ const $\;\to\;$ ray of argument $v$",
                color=MUTED).scale(0.55),
            Tex(r"$u=$ const $\;\to\;$ semicircle of radius $e^{u}$",
                color=MUTED).scale(0.55),
        ).arrange(DOWN, buff=0.17)
        uhp_note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(uhp_note, shift=0.2 * UP), run_time=0.9)
        self.wait(2.2)
        self.play(FadeOut(uhp_note), run_time=0.6)

        # ---------------------------------------- 3. Cayley onto the disk
        cay_lab = MathTex(r"\zeta=\frac{z-i}{z+i}", color=WBLUE).scale(0.9)
        cay_lab.next_to(exp_lab, DOWN, buff=0.45).align_to(exp_lab, LEFT)

        rim = disk_rim()

        self.play(FadeIn(cay_lab), run_time=0.7)

        t2 = ValueTracker(0.0)
        live2 = always_redraw(lambda: picture(t2.get_value(), uhp_pt, disk_pt))
        self.remove(live)
        self.add(live2)
        self.play(t2.animate.set_value(1.0), FadeOut(axis),
                  run_time=3.6, rate_func=smooth)
        self.play(FadeIn(rim), run_time=0.8)
        self.wait(0.5)

        disk_note = VGroup(
            Tex(r"the two edges $\to$ the two halves of $|\zeta|=1$",
                color=MUTED).scale(0.55),
            Tex(r"the midline $v=\pi/2\ \to\ $ the diameter",
                color=MUTED).scale(0.55),
            Tex(r"the little squares stay square: that is conformality",
                color=GOLD).scale(0.55),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        disk_note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(disk_note, shift=0.2 * UP), run_time=1.0)
        self.wait(2.6)

        full = MathTex(r"\zeta=\frac{e^{\,w}-i}{e^{\,w}+i}",
                       color=INK).scale(0.85)
        full.to_corner(UR, buff=0.75).shift(0.55 * DOWN)
        self.play(Write(full), run_time=1.2)
        self.wait(1.6)

        # ------------------------------------------- 4. all three, side by side
        # An always_redraw mobject fights being faded, since its updater
        # rebuilds it at full opacity every frame; freeze it first.
        frozen = picture(1.0, uhp_pt, disk_pt)
        self.remove(live2)
        self.add(frozen)
        self.play(FadeOut(disk_note), FadeOut(rim), FadeOut(exp_lab),
                  FadeOut(cay_lab), FadeOut(full), FadeOut(frozen),
                  run_time=0.9)

        panels, captions = VGroup(), VGroup()
        for x, mapper, name, form in (
                (-4.6, strip_pt, "strip", r"w"),
                (0.0, uhp_pt, "upper half plane", r"z=e^{\,w}"),
                (4.6, disk_pt, "unit disk", r"\zeta=\tfrac{z-i}{z+i}")):
            g = picture(0.0, mapper, mapper)
            if mapper is disk_pt:
                g.add(*disk_rim())
            g.scale(0.40).move_to([x, 0.55, 0.0])
            panels.add(g)
            cap = VGroup(Tex(name, color=INK).scale(0.60),
                         MathTex(form, color=MUTED).scale(0.60))
            cap.arrange(DOWN, buff=0.20)
            cap.move_to([x, -1.55, 0.0])
            captions.add(cap)

        self.play(LaggedStart(*[FadeIn(p) for p in panels], lag_ratio=0.25),
                  run_time=1.8)
        self.play(LaggedStart(*[FadeIn(c) for c in captions], lag_ratio=0.25),
                  run_time=1.2)
        self.wait(3.0)
