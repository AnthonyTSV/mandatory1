import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Dict, Union, List
import os

CURRENT_DIR = os.path.dirname(__file__)

x, y, t = sp.symbols("x,y,t")


class Wave2D:

    def __init__(self):
        self.xij: np.ndarray = None
        self.yij: np.ndarray = None
        self.N: int = None
        self.Nt: int = None
        self.dx: float = None
        self.cfl: float = None
        self.c: float = None
        self.mx: int = None
        self.my: int = None
        self.D2x: sparse.lil_matrix = None
        self.Unp1: np.ndarray = None
        self.Un: np.ndarray = None
        self.Unm1: np.ndarray = None

    def create_mesh(self, N: int, sparse: bool = False) -> None:
        """Create 2D mesh and store in self.xij and self.yij"""
        L = 1
        x = np.linspace(0, L, N + 1)
        y = np.linspace(0, L, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij", sparse=sparse)

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self) -> float:
        """Return the dispersion coefficient"""
        kx = self.mx * np.pi
        ky = self.my * np.pi
        w_ = self.c**2 * (kx**2 + ky**2)
        return np.sqrt(w_)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> None:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Unp1, Un, Unm1 = np.zeros((3, N + 1, N + 1))
        ue = self.ue(mx, my)
        Unm1[:] = sp.lambdify((x, y, t), ue)(self.xij, self.yij, 0)
        Un[:] = Unm1[:] + 0.5 * (self.c * self.dt) ** 2 * (
            self.D2x @ Unm1 + Unm1 @ self.D2x.T
        )
        self.Unp1 = Unp1
        self.Un = Un
        self.Unm1 = Unm1

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u0 = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        u = u - u0
        return self.dx * np.sqrt(np.sum(u**2))

    def apply_bcs(self) -> None:
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, 0] = 0
        self.Unp1[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ) -> Union[Dict[int, np.ndarray], Tuple[float, float]]:
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.dx = 1 / N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.D2x = self.D2(N) / self.dx**2
        self.create_mesh(N)
        self.initialize(N, mx, my)
        data = {0: self.Unm1.copy()}
        for n in range(1, Nt):
            self.Unp1[:] = (
                2 * self.Un
                - self.Unm1
                + (self.c * self.dt) ** 2 * (self.D2x @ self.Un + self.Un @ self.D2x.T)
            )
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if n % store_data == 0:
                data[n] = self.Un.copy()
        if store_data > 0:
            return data
        elif store_data == -1:
            return (self.dx, self.l2_error(self.Un, Nt * self.dt))

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self) -> None:
        return None


def test_convergence_wave2d() -> None:
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann() -> None:
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d() -> None:
    N=999
    Nt=10
    mx=3
    my=3
    store_data=-1
    cfl=1/np.sqrt(2)
    solD = Wave2D()
    solN = Wave2D_Neumann()
    errD = solD(N, Nt, mx=mx, my=my, store_data=store_data, cfl=cfl)[1]
    errN = solN(N, Nt, mx=mx, my=my, store_data=store_data, cfl=cfl)[1]
    assert abs(errD) < 1e-12 and abs(errN) < 1e-12


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    # sol = Wave2D_Neumann()
    # plotdata = sol(N=40, Nt=170, mx=2, my=2, store_data=5, cfl=1/np.sqrt(2))
    # import matplotlib.animation as animation

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("u")
    # frames = []
    # for n, val in plotdata.items():
    #     # frame = ax.plot_wireframe(sol.xij, sol.yij, val, rstride=2, cstride=2)
    #     frame = ax.plot_surface(
    #         sol.xij,
    #         sol.yij,
    #         val,
    #         vmin=-0.5 * plotdata[0].max(),
    #         vmax=plotdata[0].max(),
    #         cmap=cm.coolwarm,
    #         linewidth=0,
    #         antialiased=False,
    #     )
    #     frames.append([frame])

    # ani = animation.ArtistAnimation(fig, frames, interval=400)
    # ani.save(
    #     CURRENT_DIR + "/wave_neumann.gif", writer="pillow", fps=5
    # )
    
