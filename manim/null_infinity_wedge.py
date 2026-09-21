"""Figure 7 of 2609.01706, built one step at a time.

The Penrose diagram of Minkowski is drawn from the actual conformal map, so
every null line really is at 45 degrees and the r = const curves really do
run from i^- to i^+. With u = t - r and v = t + r,

    T = arctan(v) + arctan(u),    X = arctan(v) - arctan(u),

and (T, X) is placed on screen with r = 0 at X = 0 on the left, i^0 at
X = pi on the right, and i^+ at T = pi. The edge T + X = pi is I^+.

The construction: a strip anchored at a cut u = u_0 of I^+ and cut off at
r = r_min, then the bulk Bondi mass evolves it to the future along I^+, then
the timelike tube theorem fills in the future of the strip's inner corner,
then the time slice axiom fills the whole wedge spacelike to the cut.

Note the region reached by the Bondi mass step is r > r_min, not r < r_min:
the Killing flow preserves r, and the operator has to land back inside the
strip, which lives at r > r_min. The figure agrees; the paper's prose at
that point does not.

Render (fast, low res):
    python -m manim -ql manim/null_infinity_wedge.py NullInfinityWedge
Render (1080p60):
    python -m manim -qh manim/null_infinity_wedge.py NullInfinityWedge
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

# \mathscr is not in manim's default preamble, and I^+ wants the script I
TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{mathrsfs}")
config.tex_template = TEX

INK = "#E8EEF7"
MUTED = "#97A3B6"
TEAL = "#2DD4BF"
TEAL_HI = "#7FF3E4"
WBLUE = "#5B9BFF"
AMBER = "#FFD166"
FILL = "#9FC4E8"

R0X = -6.0                  # the r = 0 line
WID = 3.0                   # i^0 sits this far to the right of it
HGT = 3.0                   # and i^+ this far above the centre

U0 = -0.25                  # the cut
U1 = 0.28                   # and the far edge of the strip, u_0 + delta u
# r_min is set so the strip stays roughly square: its depth is
# sqrt(2)*(3/pi)*(pi/2 - arctan(u_0 + 2 r_min)), which grows as the cut moves
# earlier, so an earlier cut needs a larger r_min to keep the box from
# stretching out towards I^+.
RMIN = 0.93


def pt(T, X):
    return np.array([R0X + X / PI * WID, T / PI * HGT, 0.0])


def uv(u, v):
    """A bulk point, from its null coordinates."""
    tu, tv = np.arctan(u), np.arctan(v)
    return pt(tv + tu, tv - tu)


def scri(u):
    """The point of I^+ at retarded time u."""
    tu = np.arctan(u)
    return pt(PI / 2 + tu, PI / 2 - tu)


def axis(u):
    """Where the outgoing ray of retarded time u leaves r = 0."""
    return uv(u, u)


def r_curve(r, u_from=None, u_to=None, n=160):
    """The timelike curve r = const, or the part of it between two cuts."""
    lo = -PI / 2 + 1e-3 if u_from is None else np.arctan(u_from)
    hi = PI / 2 - 1e-3 if u_to is None else np.arctan(u_to)
    return [uv(np.tan(th), np.tan(th) + 2.0 * r)
            for th in np.linspace(lo, hi, n)]


def scri_arc(u_from, u_to=None, n=90):
    hi = PI / 2 - 1e-3 if u_to is None else np.arctan(u_to)
    return [scri(np.tan(th))
            for th in np.linspace(np.arctan(u_from), hi, n)]


IP = pt(PI, 0.0)            # i^+
I0 = pt(0.0, PI)            # i^0
IM = pt(-PI, 0.0)           # i^-

A = scri(U0)                         # the cut itself
B = scri(U1)                         # the other end of the strip on I^+
C = uv(U1, U1 + 2.0 * RMIN)          # inner corner at u_0 + delta u
D = uv(U0, U0 + 2.0 * RMIN)          # inner corner at the cut
Q0, Q1 = axis(U0), axis(U1)          # where the two rays leave r = 0

# where the past-directed ingoing ray from D reaches r = 0; this is the
# corner of I^+(D), which is what the timelike tube theorem hands us
E = np.array([R0X, D[1] + (D[0] - R0X), 0.0])


def shade(points, opacity=0.22):
    m = VMobject(stroke_width=0, fill_color=FILL, fill_opacity=opacity)
    m.set_points_as_corners(list(points) + [points[0]])
    return m


class NullInfinityWedge(Scene):
    def construct(self):
        # ------------------------------------------- the Penrose diagram
        edges = VGroup(
            Line(IM, IP, color=INK, stroke_width=3.0),      # r = 0
            Line(IP, I0, color=INK, stroke_width=3.0),      # I^+
            Line(I0, IM, color=INK, stroke_width=3.0),      # I^-
        )

        # high on the edge, clear of the squiggles, which only mark u < u_0
        lab_scri = MathTex(r"\mathscr{I}^{+}", color=INK).scale(0.78)
        lab_scri.move_to(scri(2.5) + 0.52 * normalize(np.array([1.0, 1.0, 0.0])))
        lab_r0 = MathTex(r"r=0", color=MUTED).scale(0.62)
        lab_r0.move_to([R0X - 0.56, -1.55, 0.0])
        lab_ip = MathTex(r"i^{+}", color=MUTED).scale(0.68)
        lab_ip.move_to(IP + np.array([-0.38, 0.30, 0.0]))
        lab_i0 = MathTex(r"i^{0}", color=MUTED).scale(0.68)
        lab_i0.move_to(I0 + np.array([0.40, -0.26, 0.0]))

        self.play(Create(edges), run_time=1.8)
        self.play(FadeIn(lab_scri), FadeIn(lab_r0), FadeIn(lab_ip),
                  FadeIn(lab_i0), run_time=1.0)
        self.wait(0.6)

        # ------------------------------------- the two constant-u rays
        ray0 = DashedLine(Q0, A, color=TEAL, stroke_width=4.0,
                          dash_length=0.13, dashed_ratio=0.58)
        ray1 = DashedLine(Q1, B, color=TEAL, stroke_width=2.6,
                          dash_length=0.11, dashed_ratio=0.5)
        ray1.set_opacity(0.65)

        lab_u0 = MathTex(r"u_{0}", color=TEAL_HI).scale(0.66)
        lab_u0.move_to(A + np.array([0.36, -0.14, 0.0]))

        out = normalize(np.array([1.0, 1.0, 0.0]))
        du_arrow = DoubleArrow(A + 0.30 * out, B + 0.30 * out,
                               color=MUTED, stroke_width=2.6,
                               tip_length=0.15, buff=0.0)
        lab_du = MathTex(r"\delta u", color=MUTED).scale(0.60)
        lab_du.move_to(0.5 * (A + B) + 0.72 * out)

        self.play(Create(ray0), Create(ray1), run_time=1.4)
        self.play(FadeIn(lab_u0), GrowFromCenter(du_arrow), FadeIn(lab_du),
                  run_time=0.9)
        self.wait(0.5)

        # ------------------------------------------------ 1. the slab
        slab = shade([A, B, Q1, Q0])
        self.play(FadeIn(slab), run_time=1.0)
        self.wait(0.8)

        # ------------------------- 2. cut it off at r_min: just the strip
        rmin_full = VMobject(stroke_width=0)
        rmin_full.set_points_as_corners(r_curve(RMIN))
        rmin_curve = DashedVMobject(rmin_full, num_dashes=80,
                                    dashed_ratio=0.32)
        rmin_curve.set_stroke(color=MUTED, width=2.2, opacity=0.9)

        lab_rmin = MathTex(r"r_{\text{min}}", color=MUTED).scale(0.60)
        lab_rmin.move_to(uv(-0.55, -0.55 + 2.0 * RMIN)
                         + np.array([0.46, -0.16, 0.0]))

        strip_edge = scri_arc(U0, U1) + [C] + r_curve(RMIN, U1, U0, n=60)
        strip = shade(strip_edge)

        # the strip stays outlined for the rest of the run, as in the figure
        box = VMobject(stroke_color=INK, stroke_width=2.4, fill_opacity=0.0)
        box.set_points_as_corners(strip_edge + [strip_edge[0]])

        self.play(Create(rmin_curve), FadeIn(lab_rmin), run_time=1.4)
        self.play(FadeOut(slab), FadeIn(strip), Create(box), run_time=1.1)
        # the two rays have done their job; the box carries the cut from here
        self.play(FadeOut(ray0), FadeOut(ray1), run_time=0.7)
        self.wait(0.7)

        # ----------------------------- 3. the Bondi mass evolves the strip
        # along I^+ itself, not merely towards it: the Bondi mass translates
        # the strip up the same null direction that I^+ runs along
        mid = 0.25 * (A + B + C + D)
        along = normalize(IP - I0)
        arrow = Arrow(mid, mid + 1.45 * along, color=TEAL_HI,
                      stroke_width=4.0, buff=0.0, tip_length=0.2)
        self.play(GrowArrow(arrow), run_time=0.9)
        self.wait(0.4)

        eq_top = MathTex(
            r"a_{1} = U_{\text{B}}(-t_{0})\,a_{1}(t_{0})\,"
            r"U_{\text{B}}^{\dagger}(-t_{0}) \in "
            r"\mathcal{A}_{\delta u,\,r_{\text{min}}}(u_{0})",
            color=INK).scale(0.56)
        eq_top.move_to([2.55, 1.70, 0.0])
        where = Tex("where", color=MUTED).scale(0.58)
        where.move_to([2.55, 0.95, 0.0])
        eq_bot = MathTex(
            r"U_{\text{B}}(t)=e^{-i t \hat{M}_{\text{B}}^{(2)}"
            r"(f;u^{\prime},r^{\prime})}", color=INK).scale(0.56)
        eq_bot.move_to([2.55, 0.28, 0.0])

        lune = shade(scri_arc(U0) + r_curve(RMIN, U0)[::-1])

        rad = VGroup()
        for u in (-3.2, -2.0, -1.2):
            base = scri(u)
            n = normalize(np.array([1.0, 1.0, 0.0]))
            d = normalize(np.array([-1.0, 1.0, 0.0]))
            # 2.75 periods rather than a whole number: the transverse slope
            # is then zero at the far end, so the wiggle leaves travelling
            # along n and the head continues it without a kink
            span = 0.34

            def wig(t, base=base, n=n, d=d, span=span):
                return (base + t * n
                        + 0.055 * np.sin(t / span * 2.75 * TAU) * d)

            w = VMobject(stroke_color=AMBER, stroke_width=3.0)
            w.set_points_as_corners([wig(t)
                                     for t in np.linspace(0.0, span, 50)])
            head = Line(wig(span), wig(span) + 0.18 * n,
                        stroke_color=AMBER, stroke_width=3.0)
            head.add_tip(tip_length=0.16, tip_width=0.14)
            head.tip.set_fill(AMBER, 1.0).set_stroke(AMBER, 0.0)
            rad.add(VGroup(w, head))
        lab_rad = Tex("outgoing radiation", color=AMBER).scale(0.52)
        lab_rad.move_to([-2.10, 1.30, 0.0])

        self.play(Write(eq_top), run_time=1.3)
        self.play(FadeIn(where), Write(eq_bot), run_time=1.1)
        self.play(FadeOut(strip), FadeIn(lune),
                  LaggedStart(*[Create(w) for w in rad], lag_ratio=0.18),
                  FadeIn(lab_rad), run_time=1.8)
        self.wait(1.2)
        self.play(FadeOut(arrow), run_time=0.5)

        # ----------------------------------- 4. the timelike tube theorem
        # the step captions take the equations' place rather than stacking
        # underneath them
        eqs = VGroup(eq_top, where, eq_bot)
        tube_txt = Tex("Timelike tube theorem", color=TEAL_HI).scale(0.72)
        tube_txt.move_to([2.55, 0.95, 0.0])

        glow = rmin_curve.copy().set_stroke(color=TEAL_HI, width=5.0,
                                            opacity=1.0)
        envelope = shade([D, A] + scri_arc(U0)[1:] + [IP, E])

        self.play(FadeOut(eqs), FadeIn(tube_txt), run_time=0.9)
        self.play(FadeIn(glow), run_time=0.5)
        self.play(FadeOut(lune), FadeIn(envelope), run_time=1.5)
        self.play(FadeOut(glow), run_time=0.6)
        self.wait(1.0)

        # -------------------------------------- 5. the time slice axiom
        slice_txt = Tex("Time slice axiom", color=TEAL_HI).scale(0.72)
        slice_txt.move_to([2.55, 0.95, 0.0])

        cauchy_end = np.array([R0X, 2.35, 0.0])
        cauchy = VMobject(stroke_color=AMBER, stroke_width=4.0)
        cauchy.set_points_smoothly(
            [A, 0.5 * (A + cauchy_end) + np.array([0.0, 0.16, 0.0]),
             cauchy_end])

        self.play(FadeOut(tube_txt), FadeIn(slice_txt),
                  Create(cauchy), run_time=1.5)
        self.wait(0.8)

        wedge = shade([A, IP, Q0])
        self.play(FadeOut(envelope), FadeIn(wedge), run_time=1.5)
        self.wait(0.6)
        self.play(FadeOut(cauchy), run_time=0.7)

        # ------------------------------------- 6. the type, and the i^0 limit
        type3 = MathTex(r"\text{Type III}_{1}", color=TEAL_HI).scale(0.90)
        type3.move_to([2.55, 0.95, 0.0])
        self.play(FadeOut(slice_txt), FadeIn(type3), run_time=0.8)
        self.wait(1.5)

        # clear the annotations away first, so the slide that follows is read
        # against an uncluttered diagram
        self.play(FadeOut(rmin_curve), FadeOut(lab_rmin),
                  FadeOut(rad), FadeOut(lab_rad),
                  FadeOut(du_arrow), FadeOut(lab_du), run_time=1.1)
        self.wait(0.4)

        # Push the cut back to i^0. Parameterise it by s = 1/2 + arctan(u)/pi,
        # which runs over (0, 1) as u runs over the whole line, so s -> 0 is
        # u -> -infinity and can actually be animated. The wedge the cut
        # bounds then grows until it is the entire spacetime.
        s0 = 0.5 + np.arctan(U0) / PI
        cut_s = ValueTracker(s0)

        def cut_u():
            return np.tan(PI * (cut_s.get_value() - 0.5))

        live_wedge = always_redraw(
            lambda: shade([scri(cut_u()), IP, axis(cut_u())]))
        self.remove(wedge)
        self.add(live_wedge)

        # only moved, never recoloured, so the fade below still takes
        lab_u0.add_updater(
            lambda m: m.move_to(scri(cut_u()) + np.array([0.34, -0.16, 0.0])))

        type1 = MathTex(r"\text{Type I}_{\infty}", color=TEAL_HI).scale(0.90)
        type1.move_to([2.55, 0.95, 0.0])

        # The strip travels at fixed size and parks in the corner. Its u = u_0
        # edge runs along (1,1), which is the direction of I^- as well, so
        # carrying the corner A onto i^0 lays that whole edge flat on I^- at
        # the same moment: the box ends wedged into the vertex, touching both
        # pieces of null infinity, with nothing rescaled.
        slide = AnimationGroup(
            cut_s.animate.set_value(0.001),
            box.animate.shift(I0 - A),
            FadeOut(lab_u0),
            run_time=2.8, rate_func=linear)
        # started at 85% of the slide, so the type has already changed by the
        # time the wedge finishes filling
        swap = AnimationGroup(FadeOut(type3), FadeIn(type1), run_time=0.5)
        self.play(LaggedStart(slide, swap, lag_ratio=0.85))
        lab_u0.clear_updaters()
        self.wait(2.6)
