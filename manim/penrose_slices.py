"""One state, two slices, and then a second operator, on the Penrose diagram.

Fig. 3 of "Stringy effects in scrambling" (Shenker & Stanford, arXiv:1412.6087)
makes the point that the one-particle state W(t_4)|TFD> can be represented on
any bulk slice: its left panel reads it on a slice meeting the right boundary
at t_4, its right panel on a lower slice meeting the boundary at t_3, where the
quantum has slid down its null ray and been blueshifted. Fig. 4 then acts with
V(t_3) on that early slice, where the two quanta are spacelike separated, and
the state becomes V(t_3)W(t_4)|TFD>.

This animation runs through both. The slices are nearly horizontal, waving
gently across the diagram and pinned at the right boundary; they are drawn only
at the two ends, so that the rewind carries the wiggle down its ray and leaves
the slice it was read on behind. That is the whole point: nothing about the
state changes, only which slice we read it on.

The W quantum sits on a null ray at 45 degrees a fixed distance below the u = 0
horizon -- RAY_LIFT + W_HEIGHT - S, which is negative -- so it never crosses
it. The V quantum is placed on the early slice a little in from the boundary,
which is the stretch lying just above the v = 0 horizon.

Render (fast, low res):
    python -m manim -ql manim/penrose_slices.py PenroseSlices
Render (1080p60):
    python -m manim -qh manim/penrose_slices.py PenroseSlices
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

CENTRE = np.array([-1.5, -0.1, 0.0])   # the bifurcation surface
S = 2.5                                # half width of the diagram

# null directions: D runs along the u = 0 horizon, N across it
D = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
N = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)

W_HEIGHT = 1.25             # where t_4 sits on the right boundary
RAY_LIFT = 0.55             # how far the W ray clears that insertion; with
                            # W_HEIGHT it fixes the gap to the u = 0 horizon,
                            # RAY_LIFT + W_HEIGHT - S = -0.70, the clearance
                            # under it. That has to beat 0.9 * half + LIFT +
                            # amp: the packet lies along the near flat slice
                            # while u = 0 climbs at 45 degrees, so its far end
                            # is the part that gets close to the horizon
SLICE_AMP = 0.22            # how far the slices wave
CYCLES = 1.25               # and how often, across the width
Y_LATE = W_HEIGHT           # the late slice meets the boundary at t_4
Y_EARLY = -1.25             # the early one at t_3, well below
V_DEPTH = 0.58              # how far in from the boundary the V quantum sits
LIFT = 0.09                 # how far the packets ride above their slice
T2_HEIGHT = 1.60            # where t_2 sits on the right boundary; t_2 and
                            # t_4 are both at time ~ t, so they sit close
T2_DEPTH = 0.52             # how far in from the boundary its packet sits
T2_HALF = 0.22              # short, because the t_2 slice runs close to
                            # u = 0 and converges on it going inward
T1_HEIGHT = -1.85           # t_1, well below t_3; not lower, or the wedge
                            # its slice leaves above v = 0 gets too thin
T1_DEPTH = 0.22             # its packet, in near the boundary
T1_HALF = 0.15
T2_LIFT = 0.60              # the t_2 slice's left end is raised, so it stays
                            # above the v = 0 horizon instead of cutting
                            # across it. It cannot clear it entirely: v = 0
                            # meets the left boundary at the top corner, so
                            # any slice anchored below that starts under it
V_HALF = 0.26               # short enough to stay inside the strip of slice
                            # lying above v = 0, which the slice leaves the
                            # right boundary with and closes off further in

INK = "#E8EEF7"
MUTED = "#97A3B6"
WBLUE = "#5B9BFF"
VRED = "#FF3B3B"

X_L, X_R = CENTRE[0] - S, CENTRE[0] + S


def wavy(y, cycles=8, amp=0.10, n=400):
    """A singularity, drawn as a wave rather than a sawtooth."""
    xs = np.linspace(X_L, X_R, n)
    ys = y + amp * np.sin(2 * np.pi * cycles * (xs - X_L) / (2 * S))
    curve = VMobject(stroke_color=INK, stroke_width=4)
    curve.set_points_smoothly([np.array([x, yy, 0.0]) for x, yy in zip(xs, ys)])
    return curve


def slice_y(x, y0, lift=0.0):
    """Height of the slice pinned at `y0` on the right boundary.

    The wave vanishes at the right boundary, so `y0` is exactly where the slice
    meets it, and dips just inside -- which is what lets the descending null ray
    cross the slice a little way in from the edge rather than only at the edge.
    """
    t = (X_R - x) / (2 * S)
    return (CENTRE[1] + y0
            - SLICE_AMP * np.sin(2 * np.pi * CYCLES * t)
            + lift * t ** 2)


def slope_at(x, y0, lift=0.0, eps=1e-3):
    return (slice_y(x + eps, y0, lift) - slice_y(x - eps, y0, lift)) / (2 * eps)


def bulk_slice(y0, lift=0.0, n=300):
    xs = np.linspace(X_L, X_R, n)
    curve = VMobject(stroke_color=INK, stroke_width=4)
    curve.set_points_smoothly(
        [np.array([x, slice_y(x, y0, lift), 0.0]) for x in xs])
    return curve


def at_depth(y0, depth, lift=0.0):
    """A point on the slice, `depth` in from the right boundary."""
    x = X_R - depth
    return np.array([x, slice_y(x, y0, lift), 0.0]), slope_at(x, y0, lift)


def ray_y(x):
    """The W quantum's null ray: 45 degrees, below and left of t_4."""
    return CENTRE[1] + W_HEIGHT + RAY_LIFT - (X_R - x)


def crossing(y0, n=900):
    """Where the slice meets that ray, and the slice's slope there."""
    xs = np.linspace(X_L, X_R, n)
    gaps = [slice_y(x, y0) - ray_y(x) for x in xs]
    hit = X_R
    for i in range(n - 1):
        if gaps[i] * gaps[i + 1] <= 0.0:
            span = gaps[i + 1] - gaps[i]
            w = 0.0 if abs(span) < 1e-12 else -gaps[i] / span
            hit = xs[i] + w * (xs[i + 1] - xs[i])
            break
    return np.array([hit, slice_y(hit, y0), 0.0]), slope_at(hit, y0)


def ray_meet(start, target, lift=0.0, n=1400, reach=7.0):
    """Follow the fixed-v ray up from `start` until it meets the slice `target`.

    -N is up and to the left, which is the +u direction: a quantum with large
    p^u runs along v = const, parallel to the v = 0 horizon.
    """
    prev_t, prev_gap = None, None
    for t in np.linspace(0.0, reach, n):
        p = start - t * N
        gap = p[1] - slice_y(p[0], target, lift)
        if prev_gap is not None and prev_gap * gap <= 0.0:
            span = gap - prev_gap
            w = 0.0 if abs(span) < 1e-12 else -prev_gap / span
            return start - (prev_t + w * (t - prev_t)) * N
        prev_t, prev_gap = t, gap
    return start - 3.0 * N


def wave(centre, slope, half, phase, pulse, colour):
    """A wavepacket riding on the slice, oscillating across it.

    It is lifted clear of the slice by LIFT rather than centred on it, so the
    white slice runs underneath the wiggle instead of cutting through it.
    """
    tangent = np.array([1.0, slope, 0.0])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0], 0.0])

    k = 2.6 * 0.42 / half
    us = np.linspace(-half, half, 160)
    env = np.cos(0.5 * np.pi * us / half) ** 2
    amp = 0.14 * pulse
    base = centre + LIFT * normal
    pts = [base + u * tangent + amp * e * np.sin(2 * np.pi * k * u - phase) * normal
           for u, e in zip(us, env)]
    curve = VMobject(stroke_color=colour, stroke_width=5)
    curve.set_points_smoothly(pts)
    return curve


def tape_mark(sign=-1.0):
    """A <| <| or |> |> mark for the corner of the frame.

    sign = -1 rewinds, sign = +1 runs forwards.
    """
    def head(dx):
        return Polygon(np.array([dx, 0.15, 0.0]), np.array([dx, -0.15, 0.0]),
                       np.array([dx + sign * 0.21, 0.0, 0.0]),
                       stroke_width=0, fill_color=MUTED, fill_opacity=1.0)
    return VGroup(head(0.0), head(-sign * 0.25)).move_to([5.95, 3.15, 0])


class PenroseSlices(Scene):
    def construct(self):
        # a clock the wavepackets can wiggle against, so they keep moving
        # through the waits as well as the animations
        clock = ValueTracker(0.0)
        clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(clock)

        # ------------------------------------------------- the Penrose diagram
        corners = {k: CENTRE + np.array([sx * S, sy * S, 0.0])
                   for k, (sx, sy) in {"bl": (-1, -1), "br": (1, -1),
                                       "tl": (-1, 1), "tr": (1, 1)}.items()}

        boundaries = VGroup(
            Line(corners["bl"], corners["tl"], stroke_color=INK, stroke_width=5),
            Line(corners["br"], corners["tr"], stroke_color=INK, stroke_width=5))
        singularities = VGroup(wavy(CENTRE[1] + S), wavy(CENTRE[1] - S))
        horizons = VGroup(
            Line(corners["bl"], corners["tr"], stroke_color=MUTED, stroke_width=4),
            Line(corners["tl"], corners["br"], stroke_color=MUTED, stroke_width=4))

        # The horizon labels sit inside, in the upper half, slanted along their
        # own diagonals: that is the only part of the interior the slices and
        # the two wavepackets leave alone.
        u_label = MathTex("u=0", color=MUTED).scale(0.75)
        u_label.rotate(45 * DEGREES).move_to(CENTRE + 2.6 * D - 0.22 * N)
        v_label = MathTex("v=0", color=MUTED).scale(0.75)
        # on the far side of its diagonal, to leave the upper left clear for the
        # V quantum, which ends its trip up there
        v_label.rotate(-45 * DEGREES).move_to(CENTRE - 2.25 * N - 0.20 * D)

        self.play(Create(boundaries), Create(singularities), run_time=1.2)
        self.play(Create(horizons), Write(u_label), Write(v_label), run_time=1.0)
        self.wait(0.3)

        # ---------------------------------------------- the state on a late slice
        drop = ValueTracker(0.0)
        level = lambda: Y_LATE + drop.get_value() * (Y_EARLY - Y_LATE)

        def w_packet():
            p, slope = crossing(level())
            t = clock.get_value()
            return wave(p, slope, 0.30 - 0.06 * drop.get_value(),
                        2 * np.pi * 1.25 * t,
                        0.72 + 0.28 * np.sin(2 * np.pi * 0.8 * t), WBLUE)

        packet = always_redraw(w_packet)
        late_cut = bulk_slice(Y_LATE)

        t4_at = np.array([X_R, CENTRE[1] + W_HEIGHT, 0.0])
        t4_dot = Dot(t4_at, radius=0.075, color=WBLUE)
        t4_label = MathTex("t_4", color=WBLUE).scale(0.8)
        t4_label.next_to(t4_dot, RIGHT, buff=0.18)

        # the state itself, written under the diagram
        state = MathTex("W(t_4)", r"|\mathrm{TFD}\rangle").scale(0.9)
        state[0].set_color(WBLUE)
        state[1].set_color(INK)
        state.move_to([-0.15, -3.25, 0])

        self.play(Create(late_cut), run_time=1.1)
        self.play(GrowFromCenter(t4_dot), Write(t4_label), Write(state),
                  run_time=0.9)

        # draw the packet once as a still, then hand it over to the updater
        p0, slope0 = crossing(Y_LATE)
        seed = wave(p0, slope0, 0.30, 0.0, 1.0, WBLUE)
        self.play(Create(seed), run_time=0.8)
        self.remove(seed)
        self.add(packet)

        # the slice rests here, and the packet wiggles, for two seconds
        self.wait(2.0)

        # ------------------------------------------- rewind: only the wiggle moves
        rewind = tape_mark(-1.0)
        self.play(FadeIn(rewind), run_time=0.5)
        self.play(FadeOut(late_cut), drop.animate.set_value(1.0), run_time=2.8,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(rewind), run_time=0.5)

        # -------------------------------------- and the early slice is drawn in
        early_cut = bulk_slice(Y_EARLY)
        t3_at = np.array([X_R, CENTRE[1] + Y_EARLY, 0.0])
        t3_label = MathTex("t_3", color=VRED).scale(0.8)
        t3_label.next_to(t3_at, RIGHT, buff=0.28)

        self.play(Create(early_cut), run_time=1.1)
        self.bring_to_front(packet)
        self.play(Write(t3_label), run_time=0.5)

        # the momentum the W quantum has picked up, along its horizon
        q_end, _ = crossing(Y_EARLY)
        p_tail = q_end - 0.42 * N - 0.10 * D
        p_arrow = Arrow(p_tail, p_tail + 0.70 * D, buff=0.0,
                        stroke_color=WBLUE, stroke_width=4,
                        max_tip_length_to_length_ratio=0.28)
        p_label = MathTex("p_4^v", color=WBLUE).scale(0.75)
        p_label.next_to(p_arrow, UL, buff=0.02)

        self.play(GrowArrow(p_arrow), Write(p_label), run_time=0.8)
        self.wait(1.0)

        # -------------------------------------------------- now act with V(t_3)
        # On this slice the two quanta are spacelike separated, so V can simply
        # be applied: its own quantum lands just above the v = 0 horizon, on
        # the stretch of slice near the boundary.
        v_at, v_slope = at_depth(Y_EARLY, V_DEPTH)

        def v_packet():
            t = clock.get_value()
            return wave(v_at, v_slope, V_HALF, 2 * np.pi * 1.25 * t + 1.7,
                        0.72 + 0.28 * np.sin(2 * np.pi * 0.8 * t + 1.1), VRED)

        t3_dot = Dot(t3_at, radius=0.075, color=VRED)
        v_seed = wave(v_at, v_slope, V_HALF, 1.7, 1.0, VRED)

        v_state = MathTex("V(t_3)", color=VRED).scale(0.9)
        v_state.next_to(state, LEFT, buff=0.12)

        # Its momentum runs the other way, along v = 0: +u is up and to the
        # left. The arrow is laid parallel to that horizon, a short way above
        # it, so it hugs it the whole of its length.
        q_off = 0.52
        q_arrow = Arrow(CENTRE + 2.30 * N + q_off * D,
                        CENTRE + 1.62 * N + q_off * D, buff=0.0,
                        stroke_color=VRED, stroke_width=4,
                        max_tip_length_to_length_ratio=0.28)
        q_label = MathTex("p_3^u", color=VRED).scale(0.75)
        q_label.move_to(CENTRE + 1.55 * N + (q_off + 0.30) * D)

        self.play(GrowFromCenter(t3_dot), Create(v_seed), Write(v_state),
                  run_time=1.0)
        self.remove(v_seed)
        self.add(always_redraw(v_packet))
        self.play(GrowArrow(q_arrow), Write(q_label), run_time=0.8)
        self.wait(1.6)

        # ------------------------------------ and the W of the other state, at t_2
        # The right panel of Fig. 4: the out state has its quanta above the
        # collision rather than below it. Its slice comes in with it, and the
        # packet sits on that slice near the boundary, in the thin wedge the
        # slice leaves between itself and the u = 0 horizon.
        t2_cut = bulk_slice(T2_HEIGHT, T2_LIFT)
        t2_at = np.array([X_R, CENTRE[1] + T2_HEIGHT, 0.0])
        t2_dot = Dot(t2_at, radius=0.075, color=WBLUE)
        t2_label = MathTex("t_2", color=WBLUE).scale(0.8)
        t2_label.next_to(t2_dot, RIGHT, buff=0.18)

        w2_at, w2_slope = at_depth(T2_HEIGHT, T2_DEPTH, T2_LIFT)

        def w2_packet():
            t = clock.get_value()
            return wave(w2_at, w2_slope, T2_HALF, 2 * np.pi * 1.25 * t + 3.1,
                        0.72 + 0.28 * np.sin(2 * np.pi * 0.8 * t + 2.2), WBLUE)

        w2_seed = wave(w2_at, w2_slope, T2_HALF, 3.1, 1.0, WBLUE)
        w2_state = MathTex("W(t_2)", color=WBLUE).scale(0.9)
        w2_state.next_to(v_state, LEFT, buff=0.12)

        self.play(Create(t2_cut), GrowFromCenter(t2_dot), Write(t2_label),
                  run_time=1.1)
        w2_arrow = Arrow(CENTRE + 2.17 * D + 0.14 * N,
                         CENTRE + 1.55 * D + 0.14 * N, buff=0.0,
                         stroke_color=WBLUE, stroke_width=4,
                         max_tip_length_to_length_ratio=0.28)
        w2_plabel = MathTex("p_2^v", color=WBLUE).scale(0.75)
        w2_plabel.move_to(CENTRE + 1.86 * D + 0.55 * N)

        self.play(Create(w2_seed), Write(w2_state), run_time=0.8)
        self.remove(w2_seed)
        self.add(always_redraw(w2_packet))
        self.play(GrowArrow(w2_arrow), Write(w2_plabel), run_time=0.8)
        self.wait(1.4)

        # ------------------------------------------ V(t_1), far down the boundary
        t1_cut = bulk_slice(T1_HEIGHT)
        t1_at = np.array([X_R, CENTRE[1] + T1_HEIGHT, 0.0])
        t1_dot = Dot(t1_at, radius=0.075, color=VRED)
        t1_label = MathTex("t_1", color=VRED).scale(0.8)
        t1_label.next_to(t1_dot, RIGHT, buff=0.18)

        v1_from, v1_slope_a = at_depth(T1_HEIGHT, T1_DEPTH)
        v1_to = ray_meet(v1_from, T2_HEIGHT, T2_LIFT)
        v1_slope_b = slope_at(v1_to[0], T2_HEIGHT, T2_LIFT)

        trip = ValueTracker(0.0)

        def v1_packet():
            tau = trip.get_value()
            pos = v1_from + tau * (v1_to - v1_from)
            slope = v1_slope_a + tau * (v1_slope_b - v1_slope_a)
            t = clock.get_value()
            return wave(pos, slope, T1_HALF, 2 * np.pi * 1.25 * t + 0.6,
                        0.72 + 0.28 * np.sin(2 * np.pi * 0.8 * t + 0.4), VRED)

        v1_seed = wave(v1_from, v1_slope_a, T1_HALF, 0.6, 1.0, VRED)
        v1_state = MathTex("V(t_1)", color=VRED).scale(0.9)
        v1_state.next_to(w2_state, LEFT, buff=0.12)

        self.play(Create(t1_cut), GrowFromCenter(t1_dot), Write(t1_label),
                  run_time=1.1)
        bra = MathTex(r"\langle\mathrm{TFD}|", color=INK).scale(0.9)
        bra.next_to(v1_state, LEFT, buff=0.10)

        # Its momentum arrow waits for the trip: it belongs beside the quantum
        # where that comes to rest, in the wedge the t_2 slice leaves above the
        # v = 0 horizon. Reversed, like p_2^v.
        v1_arrow = Arrow(CENTRE - 2.25 * N + 0.12 * D,
                         CENTRE - 1.70 * N + 0.12 * D, buff=0.0,
                         stroke_color=VRED, stroke_width=4,
                         max_tip_length_to_length_ratio=0.28)
        v1_plabel = MathTex("p_1^u", color=VRED).scale(0.75)
        # above the horizon, not below it, where the v = 0 label lives
        v1_plabel.move_to(CENTRE - 1.85 * N + 0.52 * D)

        self.play(Create(v1_seed), Write(v1_state), Write(bra), run_time=0.9)
        self.remove(v1_seed)
        self.add(always_redraw(v1_packet))
        self.wait(1.2)

        # The slice it was read on goes, and the quantum runs up its own null
        # ray -- fixed v, so parallel to that horizon -- until it lands on the
        # t_2 slice, which is where the out state reads it.
        forward = tape_mark(1.0)
        self.play(FadeIn(forward), run_time=0.5)
        self.play(FadeOut(t1_cut), trip.animate.set_value(1.0), run_time=2.8,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(forward), run_time=0.5)
        self.play(GrowArrow(v1_arrow), Write(v1_plabel), run_time=0.8)
        self.wait(2.6)
