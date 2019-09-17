from manimlib.imports import *


class SchrodingerEquation(Scene):
    def construct(self):
        qm = TextMobject("Quantum Mechanics")
        self.play(FadeInFromLarge(qm, scale_factor=0.1))
        self.play(ApplyMethod(qm.to_corner, UL))

        title = TextMobject("The Schrödinger Equation")
        tise = TexMobject(r"\hat H\ket\psi = E\ket\psi")

        tise.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeInFrom(tise, UP))
        self.wait()

        tdse = TexMobject(r"\hat H\ket\psi = i\hbar\pdv{}{t}\ket\psi")
        tdse.next_to(tise, DOWN + UP)
        tdse.scale(1.2)
        self.play(Transform(tise, tdse), ApplyMethod(title.shift, UP))
        self.wait()

        self.add(tdse)
        self.remove(tise)

        full = TexMobject(
            r"\qty(-\frac{\hbar^2}{2m}\laplacian + V) \ket\psi = i\hbar\pdv{}{t}\ket\psi"
        )
        full.next_to(tdse, DOWN + UP)
        full.scale(1.4)
        self.play(Transform(tdse, full))  # , ApplyMethod(title.shift, UP))
        self.wait()

        self.add(full)
        self.remove(tdse)

        full_full = TexMobject(
            r"\sum_{i=1}^N\qty(-\frac{\hbar^2}{2m_i}\laplacian_i + V) \ket\psi = i\hbar\pdv{}{t}\ket\psi"
        )
        full_full.next_to(full, DOWN + UP)
        full_full.scale(1.7)
        self.play(Transform(full, full_full), ApplyMethod(title.shift, UP))
        self.wait()

class Newton(Scene):
    def construct(self):
        qm = TextMobject("Quantum Mechanics")
        self.play(GrowFromCenter(qm))
        self.wait()

        line = Line(
            (qm.get_corner(UL) + qm.get_corner(DL)) / 2,
            (qm.get_corner(UR) + qm.get_corner(DR)) / 2,
        )
        self.play(GrowFromEdge(line, LEFT))
        newton = TextMobject("Classical Mechanics")
        tmp = TextMobject("Quantum Mechanics ")
        tmp.shift(LEFT * 3)
        newton.next_to(tmp)

        self.play(
            ApplyMethod(qm.shift, LEFT * 3),
            ApplyMethod(line.shift, LEFT * 3),
            FadeInFrom(newton, RIGHT),
        )
        self.wait()

        self.play(
            FadeOutAndShift(qm, LEFT),
            FadeOutAndShift(line, LEFT),
            ApplyMethod(newton.to_corner, UL),
        )

        grid = Axes(center_point=ORIGIN + DOWN * 3 + LEFT * 4, x_min=-1, x_max=10)
        curve = FunctionGraph(
            lambda t: -0.25 * t ** 2 + 1, x_min=-4, x_max=4, color=YELLOW_E, opacity=0.3
        )
        ball = Dot(radius=0.3, color=MAROON_E, opacity=1)
        ball.set_fill(MAROON_D, opacity=1)
        ball.move_to(curve.points[0])

        group = VGroup(grid, curve, ball)

        self.play(ShowCreation(grid, run_time=1), GrowFromCenter(ball))
        self.wait()

        self.play(ShowCreation(curve), run_time=3)
        self.play(MoveAlongPath(ball, curve), run_time=3)

        self.wait()

        newton_sub = TextMobject("Newton's 2. Law of Motion")
        newton_eq = TexMobject(r"F = ma")
        newton_sub.move_to(ORIGIN + UP)
        newton_eq.move_to(newton_sub.get_center() + 2 * DOWN)
        newton_eq.scale(1.5)

        newton_group = VGroup(newton_sub, newton_eq)

        self.play(FadeOutAndShift(group, DOWN))
        self.play(GrowFromCenter(newton_sub), Write(newton_eq))
        self.wait()

        newton_solved = TexMobject(r"x(t) = -\frac{g}{2}t^2 + v_0t + x_0")
        arrow = Vector(direction=DOWN)
        arrow.move_to(newton_eq.get_center())
        newton_solved.next_to(arrow, DOWN)

        self.play(
            ApplyMethod(newton_group.shift, UP),
            GrowFromEdge(arrow, UP),
            GrowFromEdge(newton_solved, UP),
        )
        self.wait()

        all_content = VGroup(newton_group, arrow, newton_solved, newton)

        self.play(ApplyMethod(all_content.scale, 20), run_time=3)



class Outline(Scene):
    def construct(self):
        title = TextMobject("Outline")

        outline = TextMobject(
            r"""
        \begin{itemize}
        \item Lightning intro to quantum mechanics \\
              and variational Monte Carlo (VMC)
        \item Neural networks for VMC
        \item Results
        \item Conclusions
        \end{itemize}
        """.strip()
        )

        title.scale(1.5)
        title.shift(UP * 3)

        self.play(GrowFromCenter(title), GrowFromCenter(outline))
        self.wait()
        self.play(FadeOut(title), FadeOut(outline))


class TitleScreen(Scene):
    def construct(self):
        title = TextMobject(
            r"Learning Correlations in \\Quantum Mechanics\\with Neural Networks"
        )
        subtitle = TextMobject("Thesis Presentation, September 23 2019")
        author = TextMobject("Bendik Samseth")
        title.shift(UP)
        subtitle.next_to(title, DOWN * 6)
        author.next_to(subtitle, DOWN)

        title.scale(1.5)

        self.play(Write(title))
        self.play(Write(subtitle), Write(author))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(author))
