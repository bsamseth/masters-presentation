from manimlib.imports import *


class SchrodingerEquation(Scene):
    def construct(self):
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
        self.play(Transform(tdse, full), ApplyMethod(title.shift, UP))
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
