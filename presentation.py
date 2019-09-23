from manimlib.imports import *
import numpy as np
import re


sys.path.append(os.path.dirname(__file__))
from nncode import NetworkScene


def wait(self):
    square = Square(
        side_length=0.2, fill_color="#ff0000", fill_opacity=1, stroke_opacity=0
    )
    square.move_to([-7.1, -4, 1])
    self.add(square)
    self.wait(.1)
    self.remove(square)


class ThankYou(Scene):
    def construct(self):
        thanks = TextMobject(r"Thank you for listening!")
        self.play(Write(thanks))
        wait(self)
        # wait(self)
        self.play(FadeOut(thanks))


class FutureProspects(Scene):
    def construct(self):
        title = TextMobject("Future Prospects")

        outline = TextMobject(
            r"""
        \begin{itemize}
            \item Incorporate GPU acceleration.
            \item Investigate fermionic systems.
            \item Apply different network types and \\
                    training strategies.
        \end{itemize}
        """.strip()
        )

        title.scale(1.5)
        title.shift(UP * 2.5)
        outline.scale(0.9)
        outline.shift(DOWN * 0.8)

        self.play(GrowFromCenter(title), GrowFromCenter(outline))
        wait(self)
        self.play(FadeOut(title), FadeOut(outline))


class Conclusions(Scene):
    def construct(self):
        title = TextMobject("Conclusions")

        outline = TextMobject(
            r"""
        \begin{itemize}
            \item Neural networks capable of improving accuracy of \\existing VMC approaches.
            \item Opens the door for many new models from machine learning.
            \item Still requires trail and error to find suitable architectures.
            \item Requires significantly more computing time.
            \begin{itemize}
                \item Asymptotic time-complexity unchanged.
                \item GPU parallelization can potentially speed up significantly.
            \end{itemize}
        \end{itemize}
        """.strip()
        )

        title.scale(1.5)
        title.shift(UP * 3)
        outline.scale(0.9)
        outline.shift(DOWN * 0.8)

        self.play(GrowFromCenter(title), GrowFromCenter(outline))
        wait(self)
        self.play(FadeOut(title), FadeOut(outline))


class HeliumResults(GraphScene):
    CONFIG = {
        "x_min": 0,
        "x_max": 100,
        "y_min": 0,
        "y_max": 8,
        "graph_origin": DL,
        "y_axis_label": r"$E \ [dK/ N] (+7\,K/N)$",
        "graph_origin": ORIGIN + 2.5 * DOWN + 4 * LEFT,
        "function_color": RED_E,
        "axes_color": BLUE_E,
        "x_tick_frequency": 10,
        "x_axis_label": r"\% of iterations",
        "x_labeled_nums": range(0, 101, 10),
        "y_labeled_nums": range(0, 8),
    }

    def construct(self):
        steps, bench_data, _, dnn_data, _, sdnn_data = np.loadtxt("he-results.txt")[::5].T

        results = TextMobject(r"Results: Liquid helium (32 $^4$He atoms, 3 dimensions)")
        he_eq = TexMobject(
            r"V(\mathbf{x}) = \sum_{i<j} 4\epsilon\qty[\qty(\frac{\sigma}{r_{ij}})^{12} - \qty(\frac{\sigma}{r_{ij}})^6]"
        )
        he_base = TexMobject(
            r"\psi_\text{M}(\mathbf{x}) = \exp(-\frac{1}{2}\sum_{i<j} \qty(\frac{\beta}{r_{ij}})^5)"
        )
        he_nn = TexMobject(
            r"\psi_\text{NN}(\mathbf{x}) =\psi_\text{M}(\mathbf{x})"
            r"\times \text{NN}(\mathbf{x})"
        )

        results.to_edge(UP)
        he_eq.next_to(results, DOWN)
        he_base.move_to(ORIGIN)
        he_nn.next_to(he_base, DOWN*1.2)


        pot = TextMobject(r"Potential:")
        trad = TextMobject(r"Old:")
        new = TextMobject(r"New:")
        pot.next_to(he_eq, LEFT * 2)
        trad.next_to(he_base, LEFT * 2)
        new.next_to(he_nn, LEFT * 2)
        for i in [pot, trad, new]:
            i.set_color(BLUE_E)
            i.scale(0.8)
        new.set_color(GREEN)
        pot.set_color(GREY)


        self.play(FadeIn(results))
        self.play(GrowFromEdge(pot, LEFT), Write(he_eq))
        self.play(GrowFromEdge(trad, LEFT), Write(he_base))
        self.play(GrowFromEdge(new, LEFT), Write(he_nn))
        wait(self)
        self.play(*[FadeOut(obj) for obj in [results, he_eq, he_base, he_nn, pot, trad, new]])

        self.setup_axes(animate=True)
        wait(self)

        bench_graph = self.get_graph(lambda x: 10*(7 + bench_data[int(min(99, x))]))
        dnn_graph = self.get_graph(lambda x: 10 * (7 + sdnn_data[int(min(99, x))]))

        bench_label = self.get_graph_label(
            bench_graph, label=r"\psi_\text{M}", x_val=100, direction=UR
        )
        dnn_label = self.get_graph_label(
            dnn_graph, label=r"\psi_\text{NN}", x_val=100, direction=RIGHT
        )

        self.play(Write(bench_label))
        self.play(
            ShowCreation(bench_graph), run_time=4, rate_func=lambda x: smooth(x, 12)
        )
        wait(self)
        self.play(Write(dnn_label))
        self.play(ShowCreation(dnn_graph), run_time=7, rate_func=linear)
        wait(self)

        self.play(
            *[
                FadeOut(obj)
                for obj in [bench_graph, dnn_graph, bench_label, dnn_label, self.axes]
            ]
        )

class QDResults(GraphScene):
    CONFIG = {
        "x_min": 0,
        "x_max": 100,
        "y_min": 0,
        "y_max": 6,
        "graph_origin": DL,
        "y_axis_label": r"$-\log \abs{E-E_0}$",
        "graph_origin": ORIGIN + 2.5 * DOWN + 4 * LEFT,
        "function_color": RED_E,
        "axes_color": BLUE_E,
        "x_tick_frequency": 10,
        "x_axis_label": r"\% of iterations",
        "x_labeled_nums": range(0, 101, 10),
        "y_labeled_nums": range(0, 7),
    }

    def construct(self):
        steps, bench_data, _, dnn_data, _, sdnn_data = np.loadtxt("qd-results.txt").T

        results = TextMobject(r"Results: Quantum dots (2 electrons, 2 dimensions)")
        qd_eq = TexMobject(
            r"V(\mathbf{x}_1, \mathbf{x}_2) =  \frac{1}{2}\sum_{i=1}^N\norm{\mathbf{x}_i}^2"
            + r"+ \frac{1}{r_{12}}"
        )
        qd_base = TexMobject(
            r"\psi_\text{PJ}(\mathbf{x}_1, \mathbf{x}_2) = \exp(-\alpha \sum_{i=1}^N \norm{\mathbf{x}_i}^2)",
            r"\exp(\frac{r_{12}}{1 + \beta r_{12}})",
        )
        qd_nn = TexMobject(
            r"\psi_\text{NN}(\mathbf{x}_1, \mathbf{x}_2) =\psi_\text{PJ}(\mathbf{x}_1, \mathbf{x}_2)"
            r"\times \text{NN}(\mathbf{x}_1, \mathbf{x}_2)"
        )


        results.to_edge(UP)
        qd_eq.next_to(results, DOWN)
        qd_base.move_to(ORIGIN)
        qd_base.scale(0.9)
        qd_nn.next_to(qd_base, DOWN * 1.2)

        pot = TextMobject(r"Potential:")
        trad = TextMobject(r"Old:")
        new = TextMobject(r"New:")
        pot.next_to(qd_eq, LEFT * 2)
        trad.next_to(qd_base, LEFT * 2)
        new.next_to(qd_nn, LEFT * 2)
        for i in [pot, trad, new]:
            i.set_color(BLUE_E)
            i.scale(0.8)
        new.set_color(GREEN)
        pot.set_color(GREY)

        self.play(FadeIn(results))
        self.play(GrowFromEdge(pot, LEFT), Write(qd_eq))
        self.play(GrowFromEdge(trad, LEFT), Write(qd_base))
        self.play(GrowFromEdge(new, LEFT), Write(qd_nn))
        wait(self)
        self.play(*[FadeOut(obj) for obj in [results, qd_eq, qd_base, qd_nn, pot, trad, new]])

        self.setup_axes(animate=True)
        wait(self)

        bench_graph = self.get_graph(lambda x: -np.log10(bench_data[int(x)]))
        dnn_graph = self.get_graph(lambda x: -np.log10(dnn_data[int(x)]))
        # graph_label = self.get_graph_label(
        #     gauss, label=r"\psi(x) = \exp(\alpha x^2)", x_val=-1, direction=LEFT * 2.5
        # )

        bench_label = self.get_graph_label(
            bench_graph, label=r"\psi_\text{PJ}", x_val=100, direction=DR
        )
        dnn_label = self.get_graph_label(
            dnn_graph, label=r"\psi_\text{NN}", x_val=100, direction=DR
        )

        self.play(Write(bench_label))
        self.play(
            ShowCreation(bench_graph), run_time=4, rate_func=lambda x: smooth(x, 12)
        )
        wait(self)
        self.play(Write(dnn_label))
        self.play(ShowCreation(dnn_graph), run_time=7, rate_func=linear)
        wait(self)

        self.play(
            *[
                FadeOut(obj)
                for obj in [bench_graph, dnn_graph, bench_label, dnn_label, self.axes]
            ]
        )


class NetworkDisplay(NetworkScene):
    CONFIG = {"layer_sizes": [4, 10, 8, 1], "network_mob_config": {}}

    def make_mob(self, layer_sizes):
        network = Network(sizes=layer_sizes)
        network_mob = NetworkMobject(network, **self.network_mob_config)
        return network, mob

    def construct(self):
        self.remove(self.network_mob)

        title = TextMobject(r"Neural Networks")
        title.to_corner(UL)
        self.play(FadeIn(title))
        wait(self)
        self.play(Write(self.network_mob), run_time=3)
        wait(self)

        arrows = [Vector(direction=RIGHT) for _ in range(self.layer_sizes[0])]
        [
            a.next_to(node, LEFT)
            for a, node in zip(
                arrows,
                self.network_mob.layers.submobjects[0].submobjects[0].submobjects,
            )
        ]

        labels = [
            TexMobject(r"x_1"),
            TexMobject(r"x_2"),
            TexMobject(r"x_3"),
            TexMobject(r"x_4"),
        ]
        [l.next_to(ar, LEFT) for l, ar in zip(labels, arrows)]

        self.play(*[FadeInFrom(ar, RIGHT) for ar in arrows])
        self.play(*[Write(l) for l in labels])

        wait(self)

        outarrow = Vector(Direction=RIGHT)
        out = TexMobject(r"\text{NN}(\mathbf{x})")

        outarrow.next_to(
            self.network_mob.layers.submobjects[-1].submobjects[0].submobjects[0], RIGHT
        )
        out.next_to(outarrow, RIGHT)

        self.play(FadeInFrom(outarrow, LEFT))
        self.play(ShowCreation(out))

        wait(self)

        self.play(*[FadeOut(obj) for obj in [title, out, outarrow, self.network_mob] + labels + arrows])


class NewIdea(Scene):
    def construct(self):
        idea1 = TextMobject("Idea: Use a neural network")
        idea2 = TextMobject("to learn the missing correlations")
        idea1.move_to(UP * 2)
        idea2.next_to(idea1, DOWN)

        psi = TexMobject(
            r"\psi_\text{improved} = \psi_\text{original} \times \text{NN}(\mathbf{x}_1,\dots,\mathbf{x}_N)"
        )
        psi.next_to(idea2, DOWN * 3)
        psi.scale(1.4)

        self.play(ShowCreation(idea1))
        self.play(ShowCreation(idea2))
        self.play(Write(psi))
        wait(self)
        self.play(*[FadeOut(obj) for obj in [idea1, idea2, psi]])


class PsiDesign(Scene):
    def construct(self):
        ho = TextMobject(r"Harmonic oscillator:")
        qd = TextMobject(r"Quantum dots:")
        ho_eq = TexMobject(
            r"V(\mathbf{x}_1, \mathbf{x_2},\dots,\mathbf{x}_N) = \frac{1}{2}\sum_{i=1}^N\norm{\mathbf{x}_i}^2"
        )
        qd_eq = TexMobject(
            r"V(\mathbf{x}_1, \mathbf{x}_2,\dots,\mathbf{x}_N) =  \frac{1}{2}\sum_{i=1}^N\norm{\mathbf{x}_i}^2"
            + r"+ \sum_{i < j}\frac{1}{r_{ij}}"
        )

        ho_psi = TexMobject(
            r"\psi(\mathbf{x}_1, \mathbf{x}_2, \dots,\mathbf{x}_N) = \exp(-\frac{1}{2} \sum_{i=1}^N \norm{\mathbf{x}_i}^2)"
        )
        qd_base = TexMobject(
            r"\psi(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N) = \exp(-\alpha \sum_{i=1}^N \norm{\mathbf{x}_i}^2)"
        )
        qd_jastrow = TexMobject(r"\exp(\sum_{ij} \frac{r_{ij}}{1 + \beta r_{ij}})")
        tmp = qd_base.deepcopy()
        tmp.shift(2 * LEFT)
        qd_jastrow.next_to(tmp, RIGHT)


        ho.move_to(UP * 2 + LEFT * 4)
        ho_eq.next_to(ho, RIGHT)
        qd.move_to(UP * 2 + LEFT * 5)
        qd_eq.next_to(qd, RIGHT)


        self.play(ShowCreation(ho), Write(ho_eq))
        self.play(Write(ho_psi))
        wait(self)
        self.play(Transform(ho, qd), Transform(ho_eq, qd_eq))
        wait(self)
        self.play(Transform(ho_psi, qd_base))
        wait(self)
        self.play(ApplyMethod(ho_psi.shift, LEFT * 2))
        self.play(Write(qd_jastrow))
        wait(self)

        dobetter = TextMobject("Can we do better?")
        dobetter.move_to(DOWN * 2)
        self.play(FadeIn(dobetter))
        wait(self)

        self.play(
            *[
                FadeOut(obj)
                for obj in (dobetter, qd_jastrow, ho_psi, ho, ho_eq )
            ]
        )


class WhatToGuess(Scene):
    def construct(self):
        how = TextMobject(r"How do we know what form")
        how2 = TextMobject(r"to use for $\psi$?")
        how.move_to(UP * 2)
        how2.next_to(how, DOWN)


        psi = TexMobject(r"\psi(x) = \exp(-\alpha x^2)")
        arrow = Vector(direction=RIGHT)
        psi.color = RED_E
        psi.set_color(RED_E)

        list1 = TextMobject(
            r"""
        \begin{enumerate}
        \item Theory
        \end{enumerate}
        """.strip()
        )
        list1.next_to(how2, DOWN * 2)

        arrow.next_to(list1)
        psi.next_to(arrow, RIGHT)

        list2 = TextMobject(
            r"""
        \begin{enumerate}
        \item Theory
        \item Imagination + trial \& error
        \end{enumerate}
        """.strip()
        )
        list2.next_to(how2, DOWN * 2)

        self.play(ShowCreation(how), ShowCreation(how2))
        wait(self)
        self.play(ShowCreation(list1))
        wait(self)
        self.play(GrowFromEdge(arrow, LEFT), GrowFromEdge(psi, LEFT))
        wait(self)
        self.play(FadeOut(arrow), FadeOut(psi), Transform(list1, list2))
        wait(self)

        self.play(*[FadeOut(obj) for obj in [list1, how, how2]])


class VMC(GraphScene):
    CONFIG = {
        "x_min": -4,
        "x_max": 4,
        "y_min": 0,
        "y_max": 1.3,
        "y_axis_label": r"$\psi(x)$",
        "graph_origin": ORIGIN + 2.5 * DOWN,
        "function_color": RED_E,
        "axes_color": BLUE_E,
        "x_labeled_nums": range(-3, 4, 1),
    }

    def construct(self):

        title = [
            TextMobject(r"How do we solve Schrödinger's equation"),
            TextMobject(r"for the wave function $\ket\psi$?"),
        ]
        [t.shift(UP * (1 - i)) for i, t in enumerate(title)]
        guessing = TextMobject(r"\textbf{Educated guessing}")
        guessing.shift(DOWN)
        self.play(*[FadeIn(t) for t in title])
        wait(self)
        self.play(FadeIn(guessing))
        wait(self)

        line = Line(
            (guessing.get_corner(UL) + guessing.get_corner(DL)) / 2,
            (guessing.get_corner(UR) + guessing.get_corner(DR)) / 2,
        )
        self.play(GrowFromEdge(line, LEFT))
        vmc = TextMobject(r"\textbf{Variational Monte Carlo (VMC)}")
        vmc.next_to(guessing, DOWN)
        self.play(ShowCreation(vmc))
        wait(self)

        vmc_short = TextMobject(r"\textbf{VMC}")
        vmc_short.move_to(vmc)
        self.play(
            *[FadeOut(t) for t in title + [guessing, line]], Transform(vmc, vmc_short)
        )
        self.play(ApplyMethod(vmc.to_corner, UL), run_time=0.5),

        self.setup_axes(animate=True)
        gauss = self.get_graph(self.gaussian, self.function_color)
        graph_label = self.get_graph_label(
            gauss, label=r"\psi(x) = \exp(-\alpha x^2)", x_val=-1, direction=LEFT * 2.5
        )
        alpha_label = TexMobject(r"0.5")
        arrow = Vector(direction=DOWN)
        arrow.scale(0.7)
        arrow.next_to(graph_label, UP / 2)
        arrow.shift(RIGHT * 1.2)
        alpha_label.next_to(arrow, UP / 2)

        self.play(ShowCreation(gauss), run_time=2)
        self.play(Write(graph_label))
        self.play(FadeInFrom(alpha_label, DOWN), FadeInFrom(arrow, DOWN))
        wait(self)
        # self.play(FadeOut(alpha_label), FadeOut(arrow))
        # wait(self)

        energy = TexMobject(
            r"E = \frac{\int \psi^*\hat H\psi\dd{x}}{\int \abs{\psi}^2\dd{x}}"
        )
        energy_value = TexMobject(r" = 0.5\,\text{a.u.}")
        energy.move_to(RIGHT * 3 + UP * 1.5)
        energy_value.next_to(energy, RIGHT)
        self.play(ShowCreation(energy))
        wait(self)
        self.play(ShowCreation(energy_value))
        wait(self)

        alpha_label_5 = TexMobject(r"0.5")
        alpha_label_2 = TexMobject(r"0.2")
        alpha_label_8 = TexMobject(r"0.8")
        energy_value_5 = TexMobject(r" = 0.5\,\text{a.u.}")
        energy_value_2 = TexMobject(r" = 0.725\,\text{a.u.}")
        energy_value_8 = TexMobject(r" = 0.556\,\text{a.u.}")
        [l.move_to(alpha_label) for l in (alpha_label_5, alpha_label_2, alpha_label_8)]
        [
            l.move_to(energy_value)
            for l in (energy_value_5, energy_value_2, energy_value_8)
        ]

        self.play(
            Transform(
                gauss,
                self.get_graph(lambda x: self.gaussian(x, a=0.2), self.function_color),
            ),
            Transform(alpha_label, alpha_label_2),
            Transform(energy_value, energy_value_2),
            run_time=2,
        )
        wait(self)
        self.play(
            Transform(
                gauss,
                self.get_graph(lambda x: self.gaussian(x, a=0.8), self.function_color),
            ),
            Transform(alpha_label, alpha_label_8),
            Transform(energy_value, energy_value_8),
            run_time=2,
        )
        wait(self)
        self.play(
            Transform(
                gauss,
                self.get_graph(lambda x: self.gaussian(x, a=0.5), self.function_color),
            ),
            Transform(alpha_label, alpha_label_5),
            Transform(energy_value, energy_value_5),
            run_time=2,
        )
        wait(self)
        self.play(
            *[
                FadeOut(obj)
                for obj in [
                    alpha_label,
                    energy_value,
                    energy,
                    gauss,
                    vmc,
                    arrow,
                    graph_label,
                    self.axes,
                ]
            ]
        )

    def gaussian(self, x, a=0.5):
        return np.exp(-a * (x) ** 2)


class HarmonicOscillator(Scene):
    def play_forward(self, electrons, paths, *animations, **kwargs):
        self.play(
            *[MoveAlongPath(e, p) for e, p in zip(electrons, paths)],
            *animations,
            run_time=3,
            **kwargs
        )
        for path in paths:
            path.points = path.points[::-1]

    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # grid = Axes(center_point=ORIGIN + DOWN * 3 + LEFT * 4, x_min=-1, x_max=10)
        well = FunctionGraph(
            lambda x: 0.25 * x ** 2 - 3, x_min=-6, x_max=6, color=GREY, opacity=0.3
        )

        path = FunctionGraph(
            lambda x: 0.25 * x ** 2 - 3, x_min=-4, x_max=4, color=GREY, opacity=0.3
        )
        path_reverse = path.deepcopy()
        path_short = path.deepcopy()
        path_reverse.points = path_reverse.points[::-1]
        path_short_reverse = path_reverse.deepcopy()
        path_short.points = path.points[: 2 * len(path.points) // 5]
        path_short_reverse.points = path_reverse.points[: 2 * len(path.points) // 5]

        electron = Dot(radius=0.2, color=GREEN_E, opacity=1)
        electron2 = Dot(radius=0.2, color=GREEN_E, opacity=1)
        electron.move_to(path.points[0])
        electron2.move_to(path.points[-1])

        title = TextMobject("Harmonic Oscillator")
        eq = TexMobject(r"V(x) = \frac{1}{2}x^2")
        eq_col = TexMobject(
            r"V(x_1, x_2) = \frac{1}{2}x_1^2 + \frac{1}{2}x_2^2 + \frac{1}{\abs{x_1 - x_2}}"
        )

        title.to_edge(UP)
        eq.next_to(title, DOWN)
        eq_col.next_to(title, DOWN)

        self.play(ShowCreation(well, run_time=1))
        self.play(GrowFromCenter(electron))

        self.play_forward([electron], [path], FadeIn(title), Write(eq))
        self.play_forward([electron], [path])

        wait(self)
        self.play(GrowFromCenter(electron2), Transform(eq, eq_col))
        self.play_forward([electron, electron2], [path_short, path_short_reverse])
        self.play_forward([electron, electron2], [path_short, path_short_reverse])
        self.play_forward([electron, electron2], [path_short, path_short_reverse])
        self.play_forward([electron, electron2], [path_short, path_short_reverse])

        wait(self)
        self.play(
            *[FadeOut(obj) for obj in [electron, electron2, well, eq, eq_col, title]]
        )


class SchrodingerEquation(Scene):
    def construct(self):
        plane = NumberPlane()
        qm = TextMobject("Quantum Mechanics")
        qm.move_to(ORIGIN + 0.1 * RIGHT)
        self.play(FadeInFromLarge(plane, scale_factor=0.1))
        self.play(FadeInFromLarge(qm, scale_factor=0.1))
        wait(self)

        self.play(ApplyMethod(qm.to_corner, UL))

        title = TextMobject("The Schrödinger Equation")
        tise = TexMobject(r"\hat H\ket\psi = E\ket\psi")

        tise.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeInFrom(tise, UP))
        wait(self)

        tdse = TexMobject(r"\hat H\ket\Psi = i\hbar\pdv{}{t}\ket\Psi")
        tdse.next_to(tise, DOWN + UP)
        tdse.scale(1.2)
        self.play(Transform(tise, tdse), ApplyMethod(title.shift, UP))
        wait(self)

        # self.add(tdse)
        self.remove(tise)

        full = TexMobject(
            r"\qty(-\frac{\hbar^2}{2m}\laplacian + V) \ket\Psi = i\hbar\pdv{}{t}\ket\Psi"
        )
        full.next_to(tdse, DOWN + UP)
        full.scale(1.4)
        self.play(Transform(tdse, full))  # , ApplyMethod(title.shift, UP))
        wait(self)

        self.remove(tdse)

        full_full = TexMobject(
            r"\sum_{i=1}^N\qty(-\frac{\hbar^2}{2m_i}\laplacian_i + V) \ket\Psi = i\hbar\pdv{}{t}\ket\Psi"
        )
        full_full.next_to(full, DOWN + UP)
        full_full.scale(1.7)
        self.play(Transform(full, full_full), ApplyMethod(title.shift, UP))
        wait(self)

        self.remove(full)

        final = TexMobject(
            r"\sum_{i=1}^N\qty(-\frac{\hbar^2}{2m_i}\laplacian_i + V) \ket\psi = E\ket{\psi}"
        )
        final.next_to(full_full, DOWN + UP)
        final.scale(1.3)
        self.play(Transform(full_full, final))  # , ApplyMethod(title.shift, DOWN))
        wait(self)

        self.remove(full_full)

        wavefunc = TexMobject(r"\ket{\psi} = ?")
        arrow = Vector(direction=DOWN)
        arrow.move_to(final.get_center() + DOWN)
        wavefunc.next_to(arrow, 2 * DOWN)

        wavefunc.scale(1.3)

        self.play(
            ApplyMethod(final.shift, UP),
            GrowFromEdge(arrow, UP),
            GrowFromEdge(wavefunc, UP),
        )
        wait(self)

        self.play(
            FadeOut(final),
            FadeOut(arrow),
            FadeOut(wavefunc),
            FadeOut(title),
            FadeOut(qm),
        )


class Newton(Scene):
    def construct(self):
        qm = TextMobject("Quantum Mechanics")
        self.play(GrowFromCenter(qm))
        wait(self)

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
        wait(self)

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
        wait(self)

        self.play(ShowCreation(curve), run_time=3)
        self.play(MoveAlongPath(ball, curve), run_time=3)

        wait(self)

        newton_sub = TextMobject("Newton's 2. Law of Motion")
        newton_eq = TexMobject(r"F = ma")
        newton_sub.move_to(ORIGIN + UP)
        newton_eq.move_to(newton_sub.get_center() + 2 * DOWN)
        newton_eq.scale(1.5)

        newton_group = VGroup(newton_sub, newton_eq)

        self.play(FadeOutAndShift(group, DOWN))
        self.play(GrowFromCenter(newton_sub), Write(newton_eq))
        wait(self)

        newton_solved = TexMobject(r"x(t) = -\frac{g}{2}t^2 + v_0t + x_0")
        arrow = Vector(direction=DOWN)
        arrow.move_to(newton_eq.get_center())
        newton_solved.next_to(arrow, DOWN)

        self.play(
            ApplyMethod(newton_group.shift, UP),
            GrowFromEdge(arrow, UP),
            GrowFromEdge(newton_solved, UP),
        )
        wait(self)

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
        wait(self)
        self.play(FadeOut(title), FadeOut(outline))


class TitleScreen(Scene):
    def construct(self):
        wait(self)

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
        wait(self)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(author))

