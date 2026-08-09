"""
Decompactifying a chord diagram, then reading off the chord number.

Fig. 2 of "A Cordial Introduction to Double Scaled SYK" (Berkooz & Mamroud,
arXiv:2407.09396) draws the same object twice: a chord diagram on a circle,
and the "opening and closing chords" picture on a line. This animation shows
they are the same picture -- the second is the first with the boundary circle
cut open and unrolled -- and then drops the vertical cuts one at a time,
counting the chords each one crosses.

The construction: every point of the picture is labelled by (s, d), where s is
arclength along the boundary measured from the cut and d is depth inward from
it. The boundary is then drawn as an arc of constant curvature (1-t)/R, which
at t=0 closes into the circle of radius R and at t=1 opens into a straight
segment of the same total length. Anchors sit at d=0 and so are carried along
exactly. Only the chord profile d(s) is interpolated, from a circular arc to
the rounded-rectangle arch of the paper's right-hand panel.

The anchor positions and chord pairing are read off the figure itself (the
path data in Figures/cutting_chord_diagram.pdf). Traversing from the cut, the
pairing below opens and closes chords in the order that reproduces the chord
number sequence |0> |1> |2> |3> |2> |3> |2> |1> |0> labelled in the paper's
right-hand panel -- which is what the second half of the animation counts out.

Note that each chord here is drawn on the side of the disk *not* containing
the cut, so that it has a monotone (s, d) description and its anchors stay
pinned throughout. The crossing pattern is unaffected: two chords cross iff
their anchors interleave, which is fixed by the pairing.

Render (fast, low res):
    python -m manim -ql manim/chord_decompactify.py ChordDecompactify
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

R = 2.0                 # radius of the compactified boundary
C = 2 * np.pi * R       # its circumference, preserved by the unrolling
Y_SHIFT = -2.0          # keeps both the circle and the line on screen

INK = "#E8EEF7"
TEAL = "#2DD4BF"
CUT = "#5B9BFF"
HIT = "#FF3B3B"


# ----------------------------------------------------------------- geometry

def unroll(sd, t):
    """Map an array of (arclength, depth) pairs into the plane at time t.

    The boundary is an arc of curvature (1-t)/R through the origin, pinned at
    its own midpoint s = C/2 with a horizontal tangent. Arclength is preserved,
    so t=0 is the full circle and t=1 is a straight segment of length C.
    """
    s, d = sd[:, 0], sd[:, 1]
    one_t = 1.0 - t
    if one_t < 1e-6:
        x, y = s - C / 2, d
    else:
        rho = R / one_t
        psi = (s - C / 2) / rho
        r = rho - d
        x = r * np.sin(psi)
        y = rho - r * np.cos(psi)
    return np.stack([x, y + Y_SHIFT, np.zeros_like(x)], axis=1)


def resample(pts, n):
    """Resample a polyline to n points spaced uniformly in arclength."""
    pts = np.asarray(pts, dtype=float)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    target = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(target, cum, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def _on_circle(psi, radius=R):
    return radius * np.array([np.sin(psi), -np.cos(psi)])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def arc_chord(sa, sb, depth, n=600):
    """(s, d) profile of a circular arc joining the anchors, dipping to `depth`.

    The arc is the circle through the two anchors and the point at `depth`
    below the midpoint of the sector between them, so it bows inward like the
    chords drawn in the paper rather than running straight across.
    """
    psi_a, psi_b = (sa - C / 2) / R, (sb - C / 2) / R
    A, B = _on_circle(psi_a), _on_circle(psi_b)
    M = _on_circle(0.5 * (psi_a + psi_b), R - depth)

    # circumcentre of A, M, B
    det = 2.0 * (A[0] * (M[1] - B[1]) + M[0] * (B[1] - A[1]) + B[0] * (A[1] - M[1]))
    if abs(det) < 1e-9:                                   # collinear: straight chord
        u = np.linspace(0.0, 1.0, n)[:, None]
        P = (1 - u) * A + u * B
    else:
        sA, sM, sB = A @ A, M @ M, B @ B
        ox = (sA * (M[1] - B[1]) + sM * (B[1] - A[1]) + sB * (A[1] - M[1])) / det
        oy = (sA * (B[0] - M[0]) + sM * (A[0] - B[0]) + sB * (M[0] - A[0])) / det
        O = np.array([ox, oy])
        rad = np.linalg.norm(A - O)

        aA = np.arctan2(A[1] - oy, A[0] - ox)
        aM = np.arctan2(M[1] - oy, M[0] - ox)
        aB = np.arctan2(B[1] - oy, B[0] - ox)
        sweep, to_m = _wrap(aB - aA), _wrap(aM - aA)
        if not 0.0 < to_m / sweep < 1.0:                   # take the arc through M
            sweep -= np.copysign(2 * np.pi, sweep)
        ang = aA + np.linspace(0.0, 1.0, n) * sweep
        P = O + rad * np.stack([np.cos(ang), np.sin(ang)], axis=1)

    r = np.linalg.norm(P, axis=1)
    psi = np.unwrap(np.arctan2(P[:, 0], -P[:, 1]))
    psi += np.round((psi_a - psi[0]) / (2 * np.pi)) * 2 * np.pi
    return np.stack([C / 2 + R * psi, R - r], axis=1)


def arch(sa, sb, h, rc, m=80):
    """(s, d) profile of the rounded-rectangle arch of the paper's right panel."""
    rc = min(rc, h, 0.5 * (sb - sa))
    left = np.linspace(np.pi, np.pi / 2, m)
    right = np.linspace(np.pi / 2, 0.0, m)
    pts = [[sa, 0.0], [sa, h - rc]]
    pts += [[sa + rc + rc * np.cos(a), h - rc + rc * np.sin(a)] for a in left]
    pts += [[sb - rc + rc * np.cos(a), h - rc + rc * np.sin(a)] for a in right]
    pts += [[sb, 0.0]]
    return np.array(pts, dtype=float)


class Chord:
    """One chord, carrying both of its profiles so it can be morphed between them."""

    def __init__(self, fa, fb, depth, h, rc, n=320):
        self.sa, self.sb = fa * C, fb * C
        self.circle_profile = resample(arc_chord(self.sa, self.sb, depth), n)
        self.line_profile = resample(arch(self.sa, self.sb, h, rc), n)

    def profile(self, t):
        tau = smooth(t)
        return (1 - tau) * self.circle_profile + tau * self.line_profile

    def mobject(self, t):
        curve = VMobject(stroke_color=TEAL, stroke_width=5)
        curve.set_points_smoothly(unroll(self.profile(t), t))
        return curve

    def height_at(self, s):
        """Depth of the unrolled arch above s, or None if s is outside its span."""
        if not self.sa < s < self.sb:
            return None
        return float(np.interp(s, self.line_profile[:, 0], self.line_profile[:, 1]))


# ------------------------------------------------------------------- scene

# Anchor positions as fractions of the circumference, measured from the cut.
# Read off the path data of Figures/cutting_chord_diagram.pdf.
NODES = [0.0964, 0.2244, 0.3214, 0.4786, 0.5128, 0.6936, 0.7500, 0.9639]

# depth = how far the arc dips at t=0 (also from the figure);
# h = height of the arch at t=1, chosen to keep the two crossings visible.
CHORDS = [
    Chord(NODES[0], NODES[6], depth=1.42, h=1.55, rc=0.40),   # opens 1st, closes 7th
    Chord(NODES[1], NODES[3], depth=1.06, h=1.15, rc=0.40),   # opens 2nd, closes 4th
    Chord(NODES[2], NODES[7], depth=1.53, h=0.80, rc=0.40),   # opens 3rd, closes 8th
    Chord(NODES[4], NODES[5], depth=0.70, h=0.50, rc=0.40),   # opens 5th, closes 6th
]

# One vertical cut per gap between consecutive anchors, including the two ends.
EDGES = [0.0] + NODES + [1.0]
CUTS = [0.5 * (EDGES[i] + EDGES[i + 1]) * C for i in range(len(EDGES) - 1)]

Y_BASE = Y_SHIFT
Y_TOP = Y_SHIFT + 2.35


def boundary_mobject(t):
    s = np.linspace(0.0, C, 700)
    sd = np.stack([s, np.zeros_like(s)], axis=1)
    curve = VMobject(stroke_color=INK, stroke_width=4)
    curve.set_points_smoothly(unroll(sd, t))
    return curve


def anchor_dots(t):
    s = np.array([c.sa for c in CHORDS] + [c.sb for c in CHORDS])
    sd = np.stack([s, np.zeros_like(s)], axis=1)
    return VGroup(*[Dot(p, radius=0.07, color=TEAL) for p in unroll(sd, t)])


def cut_ticks(t):
    """Blue ticks at the two ends, which coincide at t=0 and part as it opens."""
    a = unroll(np.array([[0.0, 0.0], [C, 0.0]]), t)
    b = unroll(np.array([[0.0, 0.42], [C, 0.42]]), t)
    return VGroup(*[Line(a[i], b[i], stroke_color=CUT, stroke_width=6) for i in range(2)])


def crossings(s):
    """Heights, top first, at which a vertical cut at s meets the arches."""
    ys = [c.height_at(s) for c in CHORDS]
    return sorted([y + Y_SHIFT for y in ys if y is not None], reverse=True)


class ChordDecompactify(Scene):
    def construct(self):
        t = ValueTracker(0.0)
        g = lambda f: always_redraw(lambda: f(t.get_value()))

        diagram = VGroup(g(boundary_mobject), *[g(c.mobject) for c in CHORDS],
                         g(anchor_dots), g(cut_ticks))
        self.add(diagram)

        self.wait(0.5)
        self.play(t.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        diagram.clear_updaters()
        self.wait(0.3)

        # kets are pre-rendered once and copied, so the updaters stay cheap
        kets = {n: MathTex(rf"|{n}\rangle", color=INK).scale(0.62) for n in range(4)}

        for s in CUTS:
            x = s - C / 2
            hits = crossings(s)
            y = ValueTracker(Y_TOP)

            line = always_redraw(lambda x=x, y=y: DashedLine(
                [x, Y_TOP, 0], [x, min(y.get_value(), Y_TOP - 1e-3), 0],
                dash_length=0.09, dashed_ratio=0.55,
                stroke_color=INK, stroke_width=2.2, stroke_opacity=0.75))
            blobs = always_redraw(lambda hits=hits, x=x, y=y: VGroup(*[
                Dot([x, yc, 0], radius=0.075, color=HIT)
                for yc in hits if yc >= y.get_value()]))
            ket = always_redraw(lambda hits=hits, x=x, y=y: kets[
                sum(1 for yc in hits if yc >= y.get_value())
            ].copy().move_to([x, Y_TOP + 0.3, 0]))

            group = VGroup(line, blobs, ket)
            self.add(group)
            self.play(y.animate.set_value(Y_BASE), run_time=0.8, rate_func=linear)
            group.clear_updaters()

        self.wait(1.6)
