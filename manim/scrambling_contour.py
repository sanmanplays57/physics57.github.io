"""
The path integral contour that defines the chaos correlators.

Fig. 2 of "Stringy effects in scrambling" (Shenker & Stanford, arXiv:1412.6087)
draws the contours defining the three correlators of its section 1.1: a circle
of circumference beta -- the periodic imaginary time direction -- with folds
attached, the folds being the real time evolution that produces the W(t)
operators.

This animation builds the first of those contours up, growing the correlator on
the right of the frame one operator at a time as each insertion is placed:

    1. the Euclidean circle seen face on, labelled with its direction and its
       circumference, which then tilts to an oblique view, slides left, spins
       its slit round to the front and opens it;
    2. a V insertion on the near lip of the slit, and the bra <V;
    3. a fold rising out of that lip along Lorentzian time, carrying a W
       insertion at its tip, and turning back down to the ring;
    4. the same again, a second V and a second fold, closing the contour on the
       far lip of the slit and the correlator on <V W(t) V W(t)>_TFD.

The ring is drawn as an explicit parametrisation rather than a Circle mobject,
so that the tilt (a vertical foreshortening), the slide, the in plane rotation
and the opening (an angular gap) are all continuous parameters an updater can
drive. The folds are splines through a handful of anchors, so that every joint
between a Euclidean and a Lorentzian stretch is a bend rather than a corner.

Render (fast, low res):
    python -m manim -ql manim/scrambling_contour.py ScramblingContour
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

R = 2.0                     # radius of the Euclidean time circle, face on
SQUASH = 0.42               # vertical foreshortening once it has tilted
GAP = 74.0                  # angular width of the slit, in degrees
PHASE = -38.0               # in plane rotation, swinging the slit to the front
CENTRE = np.array([-3.3, -0.6, 0.0])   # where the tilted ring ends up

FOLD_TOP = 2.7              # height the folds reach, i.e. the real time t
HALF = 0.18                 # offset from an anchor on the ring to its column
FOLD_W = 0.24               # distance between a fold's two legs
SWING = 130.0               # how far the second fold travels, in degrees
CORR = np.array([1.7, 0.1, 0.0])       # left edge of the growing correlator

INK = "#E8EEF7"
TEAL = "#2DD4BF"
VRED = "#FF3B3B"
WBLUE = "#5B9BFF"

KET = r"\rangle_{\scriptscriptstyle\text{TFD}}"


def on_ring(theta, squash=SQUASH, centre=CENTRE):
    """A point on the foreshortened ring at angle `theta`, in radians."""
    return centre + np.array([R * np.cos(theta), R * squash * np.sin(theta), 0.0])


def tangent(theta, squash=SQUASH):
    """Unit tangent to the foreshortened ring at angle `theta`, in radians."""
    t = np.array([-R * np.sin(theta), R * squash * np.cos(theta), 0.0])
    return t / np.linalg.norm(t)


def theta_at_x(x, squash=SQUASH, centre=CENTRE):
    """The angle on the near, lower half of the ring whose point sits at `x`."""
    return -np.arccos(np.clip((x - centre[0]) / R, -1.0, 1.0))


def ring(squash, gap_deg, centre, phase_deg=0.0, n=400):
    """The contour circle, foreshortened by `squash` and slit by `gap_deg`.

    The slit is centred at `phase_deg` around the ring. Swinging it round to
    the front, rather than leaving it at the far right, is what keeps each
    fold clear of the ones rising behind it.
    """
    gap, phase = np.deg2rad(gap_deg), np.deg2rad(phase_deg)
    th = phase + np.linspace(0.5 * gap, 2 * np.pi - 0.5 * gap, n)
    pts = np.stack([R * np.cos(th), R * squash * np.sin(th), np.zeros_like(th)], axis=1)
    curve = VMobject(stroke_color=INK, stroke_width=5)
    curve.set_points_as_corners(pts + centre)
    return curve


def strand(points):
    """A smooth stretch of contour through `points`."""
    curve = VMobject(stroke_color=INK, stroke_width=5)
    curve.set_points_smoothly([np.asarray(p, dtype=float) for p in points])
    return curve


def leg_up(base, theta, col, top_y, apex, sense=1.0):
    """The rising leg of a fold: a bend at the joint, then a straight climb.

    Only the joint is curved. The contour peels off the ring along its own
    tangent, settles into the column at `col` within about half a unit, and
    from there climbs dead straight before rounding over into `apex`, the
    turning point where the W insertion sits.
    """
    # a base far from its column needs longer to settle into it, or the peel
    # off turns into a diagonal dash
    settle = max(0.30, 1.4 * abs(col - base[0]))
    curve = VMobject(stroke_color=INK, stroke_width=5)
    curve.set_points_smoothly([base, base + 0.18 * tangent(theta),
                               [col, base[1] + settle, 0],
                               [col, base[1] + 1.8 * settle, 0]])
    apex = np.asarray(apex, dtype=float)
    curve.add_line_to(np.array([col, top_y - 0.28, 0.0]))
    curve.add_cubic_bezier_curve_to(np.array([col, top_y + 0.06, 0.0]),
                                    apex - sense * np.array([0.10, 0.0, 0.0]), apex)
    return curve


def leg_down(apex, col, top_y, base, sense=1.0):
    """The falling leg of a fold: the same, in reverse.

    Off the turning point, straight down the column at `col`, and a quarter
    turn at the bottom to land back on the ring at `base`.
    """
    apex, base = np.asarray(apex, dtype=float), np.asarray(base, dtype=float)
    curve = VMobject(stroke_color=INK, stroke_width=5)
    curve.start_new_path(apex)
    curve.add_cubic_bezier_curve_to(apex + sense * np.array([0.10, 0.0, 0.0]),
                                    np.array([col, top_y + 0.06, 0.0]),
                                    np.array([col, top_y - 0.28, 0.0]))
    # The landing handle has to sit between the column and the anchor. Put it
    # on the wrong side and the control polygon doubles back, curling the
    # landing into a loop. Scaling it by `sense` keeps that true even as a
    # fold turns edge on and its two legs close up.
    settle = max(0.55, 2.5 * abs(base[0] - col))
    curve.add_line_to(np.array([col, base[1] + settle, 0.0]))
    curve.add_cubic_bezier_curve_to(np.array([col, base[1] + 0.25 * settle, 0.0]),
                                    base - sense * np.array([0.12, 0.0, 0.0]), base)
    return curve


class ScramblingContour(Scene):
    def construct(self):
        squash = ValueTracker(1.0)
        gap = ValueTracker(0.0)
        cx = ValueTracker(0.0)
        cy = ValueTracker(0.0)
        phase = ValueTracker(0.0)

        circle = always_redraw(lambda: ring(squash.get_value(), gap.get_value(),
                                            np.array([cx.get_value(),
                                                      cy.get_value(), 0.0]),
                                            phase.get_value()))

        # ---------------------------------------------------------- Euclidean
        self.play(Create(ring(1.0, 0.0, ORIGIN)), run_time=1.2)
        self.remove(*self.mobjects)
        self.add(circle)
        self.wait(0.3)

        # the direction of Euclidean time: a short arc, about a sixth of the
        # circle, riding just outside it near the top
        sweep = Arc(radius=1.17 * R, start_angle=60 * DEGREES, angle=60 * DEGREES,
                    stroke_color=TEAL, stroke_width=5)
        sweep.add_tip(tip_length=0.26)
        sweep_label = Tex("Euclidean (imaginary) time", color=TEAL).scale(0.75)
        sweep_label.next_to(sweep, UP, buff=0.28)

        circumference = Tex(r"Circumference $= \beta$", color=INK).scale(0.75)

        self.play(Create(sweep), Write(sweep_label), run_time=1.3)
        self.wait(0.4)
        self.play(Write(circumference), run_time=0.9)
        self.wait(1.2)

        self.play(FadeOut(sweep), FadeOut(sweep_label), FadeOut(circumference),
                  run_time=0.7)

        # tilt to an oblique view, slide left, spin the slit round to the
        # front, and open it
        self.play(squash.animate.set_value(SQUASH), gap.animate.set_value(GAP),
                  cx.animate.set_value(CENTRE[0]), cy.animate.set_value(CENTRE[1]),
                  phase.animate.set_value(PHASE), run_time=1.6)
        self.wait(0.5)

        # --------------------------------------------------------- the anchors
        # Three points sit on the ring inside the slit: the two V insertions
        # and the point at which the contour finally closes. They are spaced
        # evenly across the slit in x, so the two folds get equal room.
        th_near = np.deg2rad(PHASE - 0.5 * GAP)
        th_far = np.deg2rad(PHASE + 0.5 * GAP)

        # Both folds are the same narrow hairpin, laid out from the near lip at
        # a fixed pitch. Two of them do not fill the slit, and the stretch left
        # over closes in the plane of the circle, as an arc of the ring itself.
        pitch = 2 * HALF + FOLD_W
        th_mid = theta_at_x(on_ring(th_near)[0] + pitch)
        th_land = theta_at_x(on_ring(th_near)[0] + 2 * pitch)
        v1, v2 = on_ring(th_near), on_ring(th_mid)
        land, shut = on_ring(th_land), on_ring(th_far)

        # Each fold is a hairpin: up one column, over the top, down the next.
        # The W insertion sits at the turning point itself.
        cols = [v1[0] + HALF, v1[0] + HALF + FOLD_W,
                v2[0] + HALF, v2[0] + HALF + FOLD_W]
        w1 = np.array([0.5 * (cols[0] + cols[1]), FOLD_TOP + 0.17, 0.0])
        w2 = np.array([0.5 * (cols[2] + cols[3]), FOLD_TOP + 0.17, 0.0])

        corr = VGroup()

        def append(tex, colour):
            """Add one factor to the correlator on the right of the frame."""
            piece = MathTex(tex, color=colour).scale(0.95)
            if len(corr) == 0:
                piece.move_to(CORR, aligned_edge=LEFT)
            else:
                piece.next_to(corr, RIGHT, buff=0.16)
            corr.add(piece)
            return piece

        def insertion(at, colour, letter, direction):
            dot = Dot(at, radius=0.09, color=colour)
            label = MathTex(letter, color=colour).scale(0.85)
            label.next_to(dot, direction, buff=0.14)
            return dot, label

        # ------------------------------------------------------ V, then a fold
        v1_dot, v1_label = insertion(v1, VRED, "V", DOWN)
        bra = append(r"\langle", INK)
        v1_bra = append("V", VRED)
        self.play(GrowFromCenter(v1_dot), FadeIn(v1_label, shift=0.2 * DOWN),
                  Write(bra), Write(v1_bra), run_time=0.9)
        self.wait(0.6)

        # the contour leaves the circle at V and runs up in real time
        up1 = leg_up(v1, th_near, cols[0], FOLD_TOP, w1)
        self.play(Create(up1), run_time=1.3)

        # the direction of Lorentzian time, marked on a short stretch of fold
        mid = 0.5 * (v1[1] + FOLD_TOP)
        lor = Arrow([cols[0] + 0.5, mid - 0.44, 0], [cols[0] + 0.5, mid + 0.44, 0],
                    buff=0, stroke_color=TEAL, stroke_width=5,
                    max_tip_length_to_length_ratio=0.26)
        lor_label = Tex("Lorentzian (real) time", color=TEAL).scale(0.5)
        lor_label.next_to(lor, RIGHT, buff=0.18)

        self.play(Create(lor), Write(lor_label), run_time=0.9)
        self.wait(0.9)
        self.play(FadeOut(lor), FadeOut(lor_label), run_time=0.6)

        w1_dot, w1_label = insertion(w1, WBLUE, "W", UP)
        w1_corr = append("W(t)", WBLUE)
        self.play(GrowFromCenter(w1_dot), FadeIn(w1_label, shift=0.2 * UP),
                  Write(w1_corr), run_time=0.9)
        self.wait(0.7)

        # ---------------------------------------------- back down to a second V
        down1 = leg_down(w1, cols[1], FOLD_TOP, v2)
        self.play(Create(down1), run_time=1.2)

        v2_dot, v2_label = insertion(v2, VRED, "V", DOWN)
        v2_corr = append("V", VRED)
        self.play(GrowFromCenter(v2_dot), FadeIn(v2_label, shift=0.2 * DOWN),
                  Write(v2_corr), run_time=0.9)
        self.wait(0.6)

        # ------------------------------------------------------ the second fold
        up2 = leg_up(v2, th_mid, cols[2], FOLD_TOP, w2)
        self.play(Create(up2), run_time=1.2)

        w2_dot, w2_label = insertion(w2, WBLUE, "W", UP)
        w2_corr = append("W(t)", WBLUE)
        self.play(GrowFromCenter(w2_dot), FadeIn(w2_label, shift=0.2 * UP),
                  Write(w2_corr), run_time=0.9)
        self.wait(0.7)

        # --------------------------------------------- down again, and it closes
        down2 = leg_down(w2, cols[3], FOLD_TOP, land)
        self.play(Create(down2), run_time=1.2)

        # what is left of the slit closes in the plane of the circle
        closing = VMobject(stroke_color=INK, stroke_width=5)
        closing.set_points_as_corners(
            [on_ring(t) for t in np.linspace(th_land, th_far, 120)])
        ket = append(KET, INK)
        self.play(Create(closing), Write(ket), run_time=1.2)
        self.wait(1.8)

        # ----------------------------------- the first V slides half way round
        # Half the Euclidean period is the -i beta/2 continuation that turns the
        # one sided correlator into a two sided one: the insertion ends up
        # antipodal on the thermal circle, on the left boundary.
        th_start = np.rad2deg(th_near) + 360.0
        slide = ValueTracker(th_start)

        def outward(t):
            v = np.array([np.cos(t), 2.2 * np.sin(t), 0.0])
            return v / np.linalg.norm(v)

        # it is still just V while it travels; the L is earned on arrival
        travelling = VGroup(
            always_redraw(lambda: Dot(on_ring(np.deg2rad(slide.get_value())),
                                      radius=0.09, color=VRED)),
            always_redraw(lambda: MathTex("V", color=VRED).scale(0.85).move_to(
                on_ring(np.deg2rad(slide.get_value()))
                + 0.44 * outward(np.deg2rad(slide.get_value())))))
        self.remove(v1_dot, v1_label)
        self.add(travelling)

        # everything still on the right boundary says so, and the folds are
        # narrow, so the two subscripted W labels need parting
        w1_R = MathTex("W_R", color=WBLUE).scale(0.85)
        w1_R.next_to(w1_dot, UP, buff=0.14).shift(0.26 * LEFT)
        v2_R = MathTex("V_R", color=VRED).scale(0.85).next_to(v2_dot, DOWN, buff=0.14)
        w2_R = MathTex("W_R", color=WBLUE).scale(0.85)
        w2_R.next_to(w2_dot, UP, buff=0.14).shift(0.26 * RIGHT)

        line1 = MathTex(r"\langle", r"V(-i\beta/2)", "W(t)", "V", "W(t)", KET)
        for piece, colour in zip(line1, [INK, VRED, WBLUE, VRED, WBLUE, INK]):
            piece.set_color(colour)
        line1.scale(0.85).move_to([1.3, 0.55, 0], aligned_edge=LEFT)

        self.play(slide.animate.set_value(th_start - 180.0),
                  Transform(w1_label, w1_R), Transform(v2_label, v2_R),
                  Transform(w2_label, w2_R),
                  ReplacementTransform(corr, line1), run_time=2.4)

        # arrived: freeze the travelling pair and let the label take its L
        travelling.clear_updaters()
        v_left_label = MathTex("V_L", color=VRED).scale(0.85)
        v_left_label.move_to(travelling[1], aligned_edge=LEFT)
        self.play(Transform(travelling[1], v_left_label), run_time=0.5)
        self.wait(0.6)

        line2 = MathTex("=", r"\langle", "V_L", "W_R(t)", "V_R", "W_R(t)", KET)
        for piece, colour in zip(line2, [INK, INK, VRED, WBLUE, VRED, WBLUE, INK]):
            piece.set_color(colour)
        line2.scale(0.85).next_to(line1, DOWN, buff=0.5, aligned_edge=LEFT)
        line2.shift(0.3 * RIGHT)

        self.play(Write(line2), run_time=1.3)
        self.wait(1.2)

        # ---------------------------- the second fold slides round as well
        # Adding +i beta/2 to the second W carries its fold round the thermal
        # circle to the left boundary. It goes anticlockwise along the ring
        # rather than across the middle, which keeps the cyclic order V W V W
        # intact and keeps it clear of the first fold. It stops SWING short of
        # a half turn, so that its leading leg comes to rest just before V_L
        # rather than sweeping over it. On the way it turns edge on, its two
        # legs closing up and reopening mirrored, exactly as a ribbon rotating
        # away from the viewer does.
        pitch_x = land[0] - v2[0]
        swing = ValueTracker(0.0)

        def hairpin(delta):
            # The legs are held FOLD_W apart wherever the fold sits, so it stays
            # as narrow as the first one. Its feet still spread and close with
            # the ring's foreshortening, so the base flares a little once the
            # fold reaches the top of the ellipse, and shuts entirely edge on.
            a, b = on_ring(th_mid + delta), on_ring(th_land + delta)
            lean = float(np.clip((b[0] - a[0]) / pitch_x, -1.0, 1.0))
            mid_x = 0.5 * (a[0] + b[0])
            ca, cb = mid_x - 0.5 * FOLD_W * lean, mid_x + 0.5 * FOLD_W * lean
            apex = np.array([0.5 * (ca + cb), FOLD_TOP + 0.17, 0.0])
            return VGroup(leg_up(a, th_mid + delta, ca, FOLD_TOP, apex, lean),
                          leg_down(apex, cb, FOLD_TOP, b, lean)), apex

        fold2 = always_redraw(lambda: hairpin(swing.get_value())[0])
        w2_dot_l = always_redraw(
            lambda: Dot(hairpin(swing.get_value())[1], radius=0.09, color=WBLUE))
        # In transit it belongs to neither boundary, so it drops its subscript
        # and only picks up the L once it has come to rest. It also rides a
        # little higher, to clear the first fold's label as it passes behind.
        w2_label_l = always_redraw(
            lambda: MathTex("W", color=WBLUE).scale(0.85).next_to(
                hairpin(swing.get_value())[1], UP, buff=0.42))

        # The ring is redrawn every frame with a second gap that travels with
        # the fold, so the fold carries its own opening along rather than
        # sliding over an intact circle, and the arc simply grows back in
        # behind it. The two gaps are the bases of the two folds: one fixed
        # under the first, one moving under the second.
        def ring_arcs(delta):
            arcs = VGroup()
            for a, b in ((th_mid, th_mid + delta),
                         (th_land + delta, th_near + 2 * np.pi)):
                if b - a > 1e-3:
                    arc = VMobject(stroke_color=INK, stroke_width=5)
                    arc.set_points_as_corners(
                        [on_ring(t) for t in np.linspace(a, b, 400)])
                    arcs.add(arc)
            return arcs

        travelling_ring = always_redraw(lambda: ring_arcs(swing.get_value()))

        self.remove(circle, closing, up2, down2, w2_dot, w2_label)
        self.add(travelling_ring, fold2, w2_dot_l, w2_label_l)
        # the ring has just been re-added, so lift everything else back over it
        self.bring_to_front(up1, down1, w1_dot, w1_label, v2_dot, v2_label,
                            travelling, fold2, w2_dot_l, w2_label_l)

        line1b = MathTex(r"\langle", r"V(-i\beta/2)", "W(t)", "V",
                         r"W(t+i\beta/2)", KET)
        for piece, colour in zip(line1b, [INK, VRED, WBLUE, VRED, WBLUE, INK]):
            piece.set_color(colour)

        line2b = MathTex("=", r"\langle", "V_L", "W_R(t)", "V_R", "W_L(t)", KET)
        for piece, colour in zip(line2b, [INK, INK, VRED, WBLUE, VRED, WBLUE, INK]):
            piece.set_color(colour)

        # line one has grown a second imaginary time; keep the pair on screen
        block = VGroup(line1b, line2b).scale(0.85)
        if block.width > 5.9:
            block.scale(5.9 / block.width)
        line1b.move_to([0.9, 0.55, 0], aligned_edge=LEFT)
        line2b.next_to(line1b, DOWN, buff=0.5, aligned_edge=LEFT).shift(0.3 * RIGHT)

        # both lines follow the fold together: the moment the first picks up
        # the +i beta/2, the second turns that W into a W_L
        self.play(swing.animate.set_value(np.deg2rad(SWING)),
                  ReplacementTransform(line1, line1b),
                  ReplacementTransform(line2, line2b), run_time=2.8)

        # and only now does that W call itself left
        for m in (fold2, w2_dot_l, w2_label_l):
            m.clear_updaters()
        # the fold has come to rest on the far side of the first one, so the
        # two labels want parting the other way round now
        w2_final = MathTex("W_L", color=WBLUE).scale(0.85)
        w2_final.next_to(w2_dot_l, UP, buff=0.14).shift(0.26 * LEFT)
        w1_R2 = MathTex("W_R", color=WBLUE).scale(0.85)
        w1_R2.next_to(w1_dot, UP, buff=0.14).shift(0.26 * RIGHT)

        self.play(Transform(w2_label_l, w2_final), Transform(w1_label, w1_R2),
                  run_time=1.3)
        self.wait(2.2)
