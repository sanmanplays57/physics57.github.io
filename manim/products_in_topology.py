"""Products in topology: Cartesian, wedge, smash, built one from the last.

S^1 x S^1 is the torus: hang a whole second circle off every point of the
first. S^1 v S^1 is the figure eight: glue the two circles at a single point.
S^1 ^ S^1 is the quotient of the one by the other.

The animation says that in three panels, left to right, each one built in the
open space before being pushed aside to make room for the next.

The camera is set at theta = -90 so that the 3D x axis runs left to right on
screen: panels are then just a shift in x, and a copy can be carried from one
panel to another without the projection fighting it. The torus spins about its
own axis rather than the camera orbiting, so the panels stay put.

The fibre over the base point at angle theta is centred a height r above it,
so it stands on the base circle rather than being threaded by it. Their union
is a torus of major radius R and minor radius r sitting on the plane of the
base circle, which is what the closing surface is.

Render (fast, low res):
    python -m manim -ql manim/products_in_topology.py Products
Render (1080p60):
    python -m manim -qh manim/products_in_topology.py Products
"""

from pathlib import Path

import numpy as np
from manim import *

config.background_color = "#0b0e1a"
config.media_dir = str(Path(__file__).resolve().parent / "media")

R = 2.2                     # radius of the base circle, lying flat
r = 0.75                    # radius of the fibre circles
FIBRES = 60                 # how many fibres close the torus up
SPIN = 0.55                 # radians per second the torus turns

PANEL = 4.65                # how far apart the panel centres sit, in x
SHRINK = 0.50               # what a panel's contents shrink to

INK = "#E8EEF7"
MUTED = "#97A3B6"
TEAL = "#2DD4BF"
WBLUE = "#5B9BFF"
GLASS = "#0e1626"           # the dark, near black torus shell
MESH = "#5f7fd0"            # and its faint mesh

# The mesh has to be much finer at full resolution or the torus reads as
# jagged, so it follows the render quality rather than being set by hand.
MESH_RES = (40, 16) if config.pixel_height <= 480 else (120, 48)

MAIN_R = 1.95               # the torus once the smash panel takes over
MAIN_r = 0.68

CART_X = -1.5               # where the first construction is built
WEDGE_X = 0.9               # and the second, in the space left over

CART = r"X \times Y = \{\,(x,y) : x \in X,\ y \in Y\,\}"
WEDGE = r"X \vee Y = (X \sqcup Y)\,/\,(x_0 \sim y_0)"
SMASH = r"X \wedge Y = (X \times Y)\,/\,(X \vee Y)"


def base_point(theta):
    return np.array([R * np.cos(theta), R * np.sin(theta), 0.0])


def fibre(theta, colour=WBLUE, width=3.5):
    """The circle standing on the base point at angle `theta`."""
    radial = np.array([np.cos(theta), np.sin(theta), 0.0])
    return ParametricFunction(
        lambda phi: ((R + r * np.cos(phi)) * radial
                     + (r + r * np.sin(phi)) * OUT),
        t_range=[0.0, TAU], color=colour, stroke_width=width)


def torus_point(u, v):
    """The surface those fibres sweep out."""
    return np.array([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u),
                     r + r * np.sin(v)])


def make_torus(major=R, minor=r, cx=0.0, zc=None, profile=None,
               stroke=0.45):
    """The torus, with an optional profile thinning the tube as u goes round."""
    if zc is None:
        zc = minor

    def point(u, v):
        rho = minor if profile is None else minor * profile(u)
        return np.array([cx + (major + rho * np.cos(v)) * np.cos(u),
                         (major + rho * np.cos(v)) * np.sin(u),
                         zc + rho * np.sin(v)])
    return Surface(point, u_range=[0.0, TAU], v_range=[0.0, TAU],
                   resolution=MESH_RES, checkerboard_colors=False,
                   fill_color=GLASS, fill_opacity=0.9,
                   stroke_color=MESH, stroke_width=stroke)


# the direction the camera sits in, for phi and theta below
CAM = np.array([0.0, -np.sin(64 * DEGREES), np.cos(64 * DEGREES)])


def shaded_loop(point_at, normal_at, colour, width=5.0, n=320, mix=1.0,
                alpha=1.0):
    """A closed curve on the torus, faded where the torus is in front of it.

    Manim's renderer sorts whole mobjects by depth, so a curve is drawn either
    wholly in front of the surface or wholly behind it. Cutting the curve into
    short pieces and setting each one's opacity from whether the surface faces
    the camera there gets the occlusion right: the near inner wall goes dark,
    the far one stays lit, which is what you see looking into a real torus.
    """
    segs = VGroup()
    ts = np.linspace(0.0, TAU, n + 1)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        lit = float(np.dot(normal_at(0.5 * (t0 + t1)), CAM))
        # smoothstep rather than a hard clip, so the curve fades into the
        # surface instead of stepping into it
        e = float(np.clip((lit + 0.60) / 1.20, 0.0, 1.0))
        segs.add(Line(point_at(t0), point_at(t1), stroke_color=colour,
                      stroke_width=width,
                      stroke_opacity=alpha * ((1.0 - mix) + mix * (
                          0.10 + 0.90 * e * e * (3.0 - 2.0 * e)))))
    return segs


def inner_equator(colour=TEAL, width=5.0, mix=1.0, profile=None, shrink=1.0,
                  alpha=1.0):
    """The circle round the hole, sitting on the inside wall.

    Contract it any further and it leaves the surface altogether, which is the
    point of putting it there rather than on the outer equator.
    """
    def rad(u):
        thin = MAIN_r if profile is None else MAIN_r * profile(u)
        return (MAIN_R - thin - 0.03) * shrink

    return shaded_loop(
        lambda u: np.array([rad(u) * np.cos(u), rad(u) * np.sin(u), MAIN_r]),
        lambda u: np.array([-np.cos(u), -np.sin(u), 0.0]), colour, width,
        mix=mix, alpha=alpha)


def meridian(a, colour=WBLUE, width=5.0, mix=1.0, shrink=1.0, alpha=1.0):
    """The circle round the tube, at angle `a` around the torus."""
    tube = (MAIN_r + 0.03) * shrink
    return shaded_loop(
        lambda v: np.array([(MAIN_R + tube * np.cos(v)) * np.cos(a),
                            (MAIN_R + tube * np.cos(v)) * np.sin(a),
                            MAIN_r + tube * np.sin(v)]),
        lambda v: np.array([np.cos(v) * np.cos(a), np.cos(v) * np.sin(a),
                            np.sin(v)]), colour, width, mix=mix, alpha=alpha)


PINCH = np.array([MAIN_R, 0.0, MAIN_r])   # where the meridian collapses


def taper(u, width=1.05):
    """One over the torus, except near u = 0, where it closes to nothing.

    Kept local on purpose: a profile that thins the whole tube makes the
    entire torus lurch, when all that should happen is one meridian closing.
    """
    d = abs((u + PI) % TAU - PI)
    t = float(np.clip(d / width, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def make_pinched():
    """The same torus, with the tube shut to a point at one meridian only."""
    # no mesh: it has already been dropped by the time this is reached, and a
    # Transform would otherwise pull the target's stroke back in
    return make_torus(MAIN_R, MAIN_r, 0.0, zc=MAIN_r, profile=taper,
                      stroke=0)


SPHERE_r = 1.45             # the sphere the torus rounds out into


def make_sphere():
    """The torus with its longitude shrunk to nothing, which is a sphere.

    Take the major radius to zero and the surface closes up into a sphere of
    the tube's radius: no quotient map, no warping, just the hole closing.
    The tube is widened as it goes so the sphere comes out a useful size.
    """
    # Exactly zero: the shape is then a true sphere, which is what matters,
    # even though every face gets drawn twice over and the copies fight in the
    # depth sort. Stopping short instead leaves a dimple, which is worse -- a
    # cross-fade can hide patchy shading but not the wrong silhouette.
    return make_torus(0.0, SPHERE_r, 0.0, zc=MAIN_r, stroke=0)


def make_shell():
    """The same sphere, but covered once.

    Taking the major radius to zero leaves the tube angle running twice round
    the sphere, so every face is drawn twice and the two copies fight in the
    depth sort, which shows up as holes in the shell. This is the same surface
    on an ordinary sphere grid, to swap in once the collapse has landed.
    """
    def point(q, a):
        return np.array([SPHERE_r * np.sin(q) * np.cos(a),
                         SPHERE_r * np.sin(q) * np.sin(a),
                         MAIN_r + SPHERE_r * np.cos(q)])
    return Surface(point, u_range=[0.0, PI], v_range=[0.0, TAU],
                   resolution=MESH_RES, checkerboard_colors=False,
                   fill_color=GLASS, fill_opacity=0.9, stroke_width=0)


def flat_circle(centre, rad, colour=TEAL, width=4.5):
    """A circle lying in a horizontal plane: a longitude of the torus."""
    c = np.asarray(centre, dtype=float)
    return ParametricFunction(
        lambda t: c + rad * np.array([np.cos(t), np.sin(t), 0.0]),
        t_range=[0.0, TAU], color=colour, stroke_width=width)


def spin_through(mob, tracker):
    """Keep `mob` turning while an animation is rewriting its points.

    An animation sets the mobject's points from scratch every frame, which
    throws away whatever an incremental updater did on the frame before, so a
    torus being moved appears to stop dead. Applying the whole accumulated
    angle each frame instead survives that. Only good during an animation --
    outside one it would compound.
    """
    mob.add_updater(
        lambda m: m.rotate(tracker.get_value(), axis=OUT,
                           about_point=m.get_center()))
    return mob


def spinning(mob):
    """Turn a mobject about its own vertical axis, wherever it happens to be."""
    mob.add_updater(
        lambda m, dt: m.rotate(SPIN * dt, axis=OUT, about_point=m.get_center()))
    return mob


def upright_circle(centre, rad, colour=TEAL, width=4.0):
    """A circle in the xz plane, which the camera sees undistorted."""
    c = np.asarray(centre, dtype=float)
    return ParametricFunction(
        lambda t: c + rad * np.array([np.cos(t), 0.0, np.sin(t)]),
        t_range=[0.0, TAU], color=colour, stroke_width=width)


def facing(mob):
    """Stand a flat mobject up into the xz plane, where the camera faces it."""
    return mob.rotate(PI / 2, axis=RIGHT)


class Products(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=64 * DEGREES, theta=-90 * DEGREES)
        off = np.array([CART_X, 0.0, 0.0])

        # ------------------------------------------------- 1. Cartesian product
        title = Tex("Cartesian Product", color=INK).scale(0.95)
        title.move_to([CART_X, 3.35, 0.0])
        formula = MathTex(CART, color=INK).scale(0.58)
        formula.move_to([4.30, 0.55, 0.0])
        self.add_fixed_in_frame_mobjects(title, formula)
        self.play(Write(title), Write(formula), run_time=1.4)
        self.wait(0.3)

        base = ParametricFunction(base_point, t_range=[0.0, TAU],
                                  color=TEAL, stroke_width=5).shift(off)
        self.play(Create(base), run_time=1.5)
        self.wait(0.4)

        mark = Dot3D(point=base_point(0.0) + off, radius=0.07, color=WBLUE)
        first = fibre(0.0, width=5).shift(off)
        self.play(FadeIn(mark, scale=0.6), run_time=0.4)
        self.play(Create(first), run_time=1.3)
        self.wait(0.6)

        rest = [fibre(t).shift(off) for t in
                np.linspace(0.0, TAU, FIBRES, endpoint=False)[1:]]
        self.play(LaggedStart(*[Create(f) for f in rest], lag_ratio=0.035),
                  run_time=5.5)
        self.wait(0.8)

        # The wireframe hands over to the surface, base circle and all, so
        # nothing is left behind to light up against it.
        torus = make_torus().shift(off)
        wire = VGroup(base, mark, first, *rest)
        self.play(FadeOut(wire), FadeIn(torus), run_time=1.6)
        spinning(torus)
        self.wait(2.0)

        # --------------------------------------------------- 2. wedge product
        torus.clear_updaters()
        carry = ValueTracker(0.0)
        carry.add_updater(lambda m, dt: m.increment_value(SPIN * dt))
        self.add(carry)
        spin_through(torus, carry)

        left_title = Tex("Cartesian Product", color=INK).scale(0.58)
        left_title.move_to([-PANEL, 3.2, 0.0])
        left_formula = MathTex(CART, color=MUTED).scale(0.40)
        left_formula.move_to([-PANEL, -2.15, 0.0])

        wedge_title = Tex("Wedge Product", color=INK).scale(0.95)
        wedge_title.move_to([2.0, 3.35, 0.0])
        wedge_formula = MathTex(WEDGE, color=INK).scale(0.55)
        wedge_formula.move_to([5.15, 0.55, 0.0])

        self.play(torus.animate.scale(SHRINK).shift((-PANEL - CART_X) * RIGHT),
                  Transform(title, left_title),
                  Transform(formula, left_formula), run_time=1.8)
        torus.clear_updaters()
        carry.clear_updaters()
        self.remove(carry)
        spinning(torus)
        # only now, or fixing them in frame would show them during the move
        self.add_fixed_in_frame_mobjects(wedge_title, wedge_formula)
        self.play(Write(wedge_title), Write(wedge_formula), run_time=1.2)

        # two circles, one at a time, each with its own base point marked
        a, gap, h = 0.75, 0.70, 0.55
        span = a + gap
        lc = upright_circle([WEDGE_X - span, 0.0, h], a)
        rc = upright_circle([WEDGE_X + span, 0.0, h], a, colour=WBLUE)
        x0_at = np.array([WEDGE_X - gap, 0.0, h])
        y0_at = np.array([WEDGE_X + gap, 0.0, h])
        x0 = Dot3D(point=x0_at, radius=0.075, color=TEAL)
        y0 = Dot3D(point=y0_at, radius=0.075, color=WBLUE)

        # The labels stand up into the xz plane, where the camera faces them,
        # and sit in the gap between the circles rather than on either curve.
        x0_lab = facing(MathTex("x_0", color=TEAL).scale(0.6))
        x0_lab.move_to(x0_at + np.array([0.28, 0.0, -0.38]))
        y0_lab = facing(MathTex("y_0", color=WBLUE).scale(0.6))
        y0_lab.move_to(y0_at + np.array([-0.28, 0.0, -0.38]))

        self.play(Create(lc), run_time=1.0)
        self.play(FadeIn(x0, scale=0.6), Write(x0_lab), run_time=0.6)
        self.play(Create(rc), run_time=1.0)
        self.play(FadeIn(y0, scale=0.6), Write(y0_lab), run_time=0.6)
        self.wait(0.7)

        # and then glued at that one point
        self.play(lc.animate.shift(gap * RIGHT), x0.animate.shift(gap * RIGHT),
                  rc.animate.shift(gap * LEFT), y0.animate.shift(gap * LEFT),
                  FadeOut(x0_lab), FadeOut(y0_lab), run_time=1.6)
        self.wait(1.2)

        # --------------------------------------------------- 3. smash product
        wedge = VGroup(lc, rc, x0, y0)

        mid_title = Tex("Wedge Product", color=INK).scale(0.58)
        mid_title.move_to([0.0, 3.2, 0.0])
        mid_formula = MathTex(WEDGE, color=MUTED).scale(0.40)
        mid_formula.move_to([0.0, -2.15, 0.0])

        smash_title = Tex("Smash Product", color=INK).scale(0.58)
        smash_title.move_to([PANEL, 3.2, 0.0])
        smash_formula = MathTex(SMASH, color=INK).scale(0.46)
        smash_formula.move_to([PANEL, -2.15, 0.0])

        self.play(wedge.animate.scale(SHRINK).shift(WEDGE_X * LEFT),
                  Transform(wedge_title, mid_title),
                  Transform(wedge_formula, mid_formula), run_time=1.6)
        self.add_fixed_in_frame_mobjects(smash_title, smash_formula)
        self.play(Write(smash_title), Write(smash_formula), run_time=1.2)
        self.wait(0.4)

        # a copy of the torus is carried across into the right hand panel
        travelling = make_torus().scale(SHRINK).shift(PANEL * LEFT)
        self.add(travelling)
        ferry = ValueTracker(0.0)
        ferry.add_updater(lambda m, dt: m.increment_value(SPIN * dt))
        self.add(ferry)
        spin_through(travelling, ferry)
        self.play(travelling.animate.shift(2 * PANEL * RIGHT), run_time=2.4)
        travelling.clear_updaters()
        ferry.clear_updaters()
        self.remove(ferry)
        spinning(travelling)
        self.wait(1.4)

        # and a copy of the figure eight follows it, into the space below
        fig8 = wedge.copy()
        self.add(fig8)
        self.play(fig8.animate.move_to([PANEL, 0.0, -1.35]), run_time=1.8)
        self.wait(0.8)

        # ------------------------------- 4. the smash panel takes the whole frame
        travelling.suspend_updating()
        main = make_torus(MAIN_R, MAIN_r, 0.0)
        self.play(FadeOut(torus), FadeOut(wedge),
                  FadeOut(title), FadeOut(formula),
                  FadeOut(wedge_title), FadeOut(wedge_formula),
                  ReplacementTransform(travelling, main),
                  smash_title.animate.scale(1.6).move_to([0.0, 3.35, 0.0]),
                  smash_formula.animate.scale(1.35).move_to([0.0, -2.55, 0.0]),
                  fig8.animate.scale(1.7).move_to([0.0, 0.0, -1.95]),
                  run_time=2.0)
        # deliberately not spinning: the pinch morphs into a target built in
        # the canonical frame, so a rotated torus would unwind as it pinched
        self.wait(1.0)

        # ----------------------- 5. the two circles are twisted onto the torus
        # One is turned flat and one left upright, which is what the torus wants:
        # a longitude runs round the hole, a meridian round the tube.
        loop_a, loop_b = fig8[0], fig8[1]
        self.play(FadeOut(fig8[2]), FadeOut(fig8[3]),
                  Rotate(loop_a, PI / 2, axis=RIGHT,
                         about_point=loop_a.get_center()),
                  run_time=1.6)
        self.wait(0.6)

        # Then dragged over the torus until each sits on it: the flat one onto
        # the inner equator, round the hole, and the upright one round the tube.
        self.play(Transform(loop_a, flat_circle([0.0, 0.0, MAIN_r],
                                                MAIN_R - MAIN_r - 0.03,
                                                colour=TEAL, width=5)),
                  Transform(loop_b, upright_circle([MAIN_R, 0.0, MAIN_r],
                                                   MAIN_r + 0.03,
                                                   colour=WBLUE, width=5)),
                  run_time=2.2)

        # Everything from here is driven by trackers and redrawn each frame,
        # rather than morphed with Transforms. The curves have to recompute
        # their own occlusion as they deform, and a Transform interpolating
        # between two already-faded copies drops them out part way.
        mixv = ValueTracker(0.0)      # how much of the occlusion is applied
        gw = ValueTracker(5.0)        # stroke width, for the glow
        morph = ValueTracker(0.0)     # longitude: round, then drawn to the pinch
        eq_shrink = ValueTracker(1.0) # longitude: collapsing to the axis
        tb_shrink = ValueTracker(1.0) # meridian: closing to a point
        eq_alpha = ValueTracker(1.0)  # and each fading out as it goes, so
        tb_alpha = ValueTracker(1.0)  # neither is left behind as a dot

        def eq_now():
            m = morph.get_value()
            return inner_equator(mix=mixv.get_value(), width=gw.get_value(),
                                 profile=lambda u: 1.0 + m * (taper(u) - 1.0),
                                 shrink=eq_shrink.get_value(),
                                 alpha=eq_alpha.get_value())

        def tb_now():
            return meridian(0.0, mix=mixv.get_value(), width=gw.get_value(),
                            shrink=tb_shrink.get_value(),
                            alpha=tb_alpha.get_value())

        equator = always_redraw(eq_now)
        tube = always_redraw(tb_now)
        self.remove(loop_a, loop_b)
        self.add(equator, tube)

        # they start fully opaque, matching what was there, and the occlusion
        # is eased in: cutting straight to it reads as a flicker
        self.play(mixv.animate.set_value(1.0), run_time=1.2)
        self.wait(0.4)

        # and glow, once
        self.play(gw.animate.set_value(11.0), run_time=0.45)
        self.play(gw.animate.set_value(5.0), run_time=0.6)
        self.wait(0.8)

        # ------------------------- 6. the mesh goes, and the meridian pinches
        # A Surface takes its mesh at build time and set_stroke will not touch
        # it, but setting the stroke on each face will. So fade the grid out by
        # driving that width down, rather than swapping the shell and having
        # the grid vanish between one frame and the next.
        grid = ValueTracker(0.45)
        faces = main.family_members_with_points()

        def fade_grid(m):
            w = grid.get_value()
            for f in faces:
                f.set_stroke(width=w)

        main.add_updater(fade_grid)
        self.play(grid.animate.set_value(0.0), run_time=1.2)
        main.clear_updaters()
        self.wait(0.4)

        caption = Tex("pinched torus", color=MUTED).scale(0.7)
        caption.move_to([0.0, -1.75, 0.0])
        self.add_fixed_in_frame_mobjects(caption)
        self.play(Write(caption), run_time=0.7)

        # The blue circle and the tube it sits on are the same circle, so they
        # close together. The longitude is drawn out to the pinch at the same
        # time, so it ends up meeting the meridian at exactly one point, which
        # is the wedge sitting inside the torus.
        self.play(Transform(main, make_pinched()),
                  morph.animate.set_value(1.0),
                  tb_shrink.animate.set_value(0.0),
                  tb_alpha.animate(run_time=2.0).set_value(0.0),
                  run_time=2.4)
        self.wait(1.0)

        # ------------------- 7. and the hole closes on that point: a sphere
        # The longitude contracts to the axis, and the hole shuts with it. It
        # finishes first: the hole is closed by the time the shell has finished
        # rounding out.
        sphere_caption = Tex("sphere", color=MUTED).scale(0.7)
        sphere_caption.move_to([0.0, -1.75, 0.0])
        self.add_fixed_in_frame_mobjects(sphere_caption)
        self.remove(sphere_caption)

        self.play(Transform(main, make_sphere()),
                  Transform(caption, sphere_caption),
                  eq_shrink.animate(run_time=1.4).set_value(0.0),
                  eq_alpha.animate(run_time=1.4).set_value(0.0),
                  run_time=2.6)
        # The collapse has to stop a hair short of degeneracy, where the shell
        # would be drawn twice over and fight itself, so its last state still
        # carries a dimple. Cross-fading into the cleanly covered sphere hides
        # that: same geometry, so only the shading changes hands.
        shell = make_shell()
        self.add(shell)
        self.play(FadeIn(shell), FadeOut(main), run_time=0.6)
        self.remove(main, equator, tube)
        main = shell

        # eased up to speed rather than snapping into motion
        rate = ValueTracker(0.0)
        main.add_updater(lambda m, dt: m.rotate(rate.get_value() * dt,
                                                axis=OUT,
                                                about_point=m.get_center()))
        self.play(rate.animate.set_value(SPIN), run_time=1.4)
        self.wait(0.8)

        # ---------------------------------- 8. all three, and what they are
        left_result = MathTex(r"S^1 \times S^1 = T^2", color=TEAL).scale(0.52)
        left_result.move_to([-PANEL, -2.85, 0.0])
        mid_result = MathTex(r"S^1 \vee S^1 = \text{figure 8}",
                             color=TEAL).scale(0.52)
        mid_result.move_to([0.0, -2.85, 0.0])
        right_result = MathTex(r"S^1 \wedge S^1 = S^2", color=TEAL).scale(0.52)
        right_result.move_to([PANEL, -2.85, 0.0])

        self.play(FadeOut(caption), FadeIn(torus), FadeIn(wedge),
                  FadeIn(title), FadeIn(formula),
                  FadeIn(wedge_title), FadeIn(wedge_formula),
                  main.animate.scale(SHRINK).shift(PANEL * RIGHT),
                  smash_title.animate.scale(1 / 1.6).move_to([PANEL, 3.2, 0.0]),
                  smash_formula.animate.scale(1 / 1.35).move_to([PANEL, -2.15, 0.0]),
                  run_time=2.0)
        # added only now: fixing a mobject in frame puts it on screen at once,
        # so adding it earlier would show all three before they are written
        self.add_fixed_in_frame_mobjects(left_result, mid_result, right_result)
        self.play(Write(left_result), Write(mid_result), Write(right_result),
                  run_time=1.6)
        self.wait(3.0)
