#!/usr/bin/env python
"""Provides Cauchy, MatrixInversion, HeatEquation classes for numeric 
analysis.

MathFunc hold a function and defines a mathematical function easier to 
use.
Cauchy compute numerical solutions of IVPs through Euler, Euler implicit
and Runge-Kutta methods. 
MatrixInversion compute Jacobi, Gauss-Seidel, SOR, Cholesky methods of
matrices inversion.
HeatEquation compute heat equation bi- and tridimensional numerical 
solutions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>. 
"""

__author__ = "Lil Tel"
__copyright__ = ""
__credits__ = ["Lil Tel"]
__license__ = "GPL3"
__version__ = "0.0.2"
__maintainer__ = "Lil Tel"
__email__ = "lil.tel@outlook.fr"
__status__ = "Alpha"

##System imports
import os
import pip
import platform
import subprocess
import sys

##Python imports
import cmath
import itertools
import functools
import math
import random
import urllib.request

from itertools import *
from random import random
from turtle import *
from typing import Any, Callable, List, Optional, Sequence, Tuple,\
    TypeVar, Union
from warnings import warn

##Macros
PLAT: str = platform.system()
YES: set = {'yes', 'Y', 'y'}
NO: set = {'no', 'N', 'n'}
ANA_VERSION = "2021.05"
PYTH: str = 'python' + {'Linux':'3 '}.get(PLAT,' ')
PIP_INSTALL: str = "-m pip install --user -q --exists-action i "
is_64bits = sys.maxsize > 2**32

##Tweaks
cd: bool = True
u_p: bool = True
np: bool = True
sc: bool = True
mtplt: bool = True
nb: bool = True
cp: bool = True

##Third-party imports

####Anaconda Installation
def import_or_install_conda() -> bool:
    try:
        __import__("conda")
        return True
    except ImportError:
        inp: str = str()
        while inp not in YES|NO:
            inp: str = input(
                'anaconda is not installed. Do you want to install it?'
                + '(yes/no)'
                )

        if inp in YES:
            while inp not in 'ma':
                inp: str = input(
                    'Do you want to install miniconda or anaconda? '
                    + '(m/a)'
                    )
            url: str = "https://repo.anaconda.com/"
            if inp == 'm': url += "miniconda/Miniconda3-latest-"
            else: url += "archive/Anaconda3-" + ANA_VERSION + "-"
            url += (
                    {
                        "Windows": "Windows", "Darwin": "MacOSX", 
                        "Linux": "Linux"
                    }[PLAT] + "-" 
                + ("x86_64" if is_64bits else "x86")
                + (".exe" if PLAT == "Windows" else ".sh")
                )
            print(url)
            urllib.request.urlretrieve(
                url, "installer." + url.split(".")[-1]
                )
            os.system({
                "Windows": "start installer.exe /S /AddToPath=1",
                "Darwin": "bash installer.sh", 
                "Linux": "bash installer.sh"
                }[PLAT])
            os.system('conda update conda')
            try:
                if nb:
                    os.system('conda install -c numba icc_rt')
                    os.system('conda update -c numba icc_rt')
            except NameError: pass

try:
    if cd: cd = import_or_install_conda()
except NameError: pass

####Pip upgrade
def upgrade_pip() -> int:
    return os.system(PYTH + PIP_INSTALL + 'pip')

try:
    if u_p: u_p = upgrade_pip()
except NameError: pass

####Matplotlib, Numba, NumPy, SciPy installation check
def import_or_install(*args) -> Tuple[bool]:
    try: 
        __import__(args[0])
        return (True,) + import_or_install(*args[1:])
    except IndexError: return ()
    except ImportError:
        while ...:
            inp: str = input(
                args[0] + ' is not installed. Do you want to install it?'
                + '(yes/no)'
                ) 
            if inp in YES|NO: break
        if inp in YES: return (
            not os.system(PYTH + PIP_INSTALL + args[0]),
            ) + import_or_install(*args[1:])
        else: return (False,) + import_or_install(*args[1:])

list_imports: List[str]= [ 'numpy', 'scipy', 'matplotlib', 'cvxpy',
                'np', 'sc', 'mtplt', 'cp'
                ]
l: int = len(list_imports)

for i in range(l//2):
    try: 
        eval(list_imports[l//2])
        list_imports.insert(l//2 - 1, list_imports.pop(0))
        list_imports.append(list_imports.pop(l//2))

    except NameError:
        exec(list_imports.pop(l//2)+'=False')
        l -= list_imports.pop(0) and 2
if l: exec(str(','.join(list_imports[l//2:])) + '= import_or_install(*'
        + str(list_imports[:l//2]) + ')'
        )

if mtplt:
    del mtplt

    import matplotlib as mtplt

    import matplotlib.animation as ani
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.image as mpimg
    import matplotlib.lines as mplns
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from matplotlib.backend_bases import MouseButton
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D

if np:
    del np

    import numpy as np

    from numpy import array as ar, diag as dg, inf as χ,\
        meshgrid as mg, pi as π, sqrt as sq, tril as L, triu as U, \
        zeros as _0
    from numpy.linalg import norm, inv as _1, solve as S

if nb:
    del nb
    
    import numba as nb

    from numba import jit, njit, prange, vectorize, int32, int64, \
        float32, float64, complex64, complex128
    from numba.experimental import jitclass
    from numba.typed import List
else:
    #https://stackoverflow.com/questions/3888158
    #https://github.com/ptooley/numbasub/blob/master/src/numbasub/nonumba.py
    def optional_arg_decorator(fn):
        @functools.wraps(fn)
        def wrapped_decorator(*args, **kwargs):
    #        is_bound_method = hasattr(args[0], fn.__name__) if args else False

    #        if is_bound_method:
    #            klass = args[0]
    #            args = args[1:]

            # If no arguments were passed...
            if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
    #            if is_bound_method:
    #                return fn(klass, args[0])
    #            else:
                    return fn(args[0])

            else:
                def real_decorator(decoratee):
    #                if is_bound_method:
    #                    return fn(klass, decoratee, *args, **kwargs)
    #                else:
                        return fn(decoratee, *args, **kwargs)
                return real_decorator
        return wrapped_decorator

    @optional_arg_decorator
    def __noop(func, *args, **kwargs):
        return(func)
        
    autojit = generated_jit = guvectorize = jit = jitclass = __noop
    njit = vectorize = __noop

    b1 = bool_ = boolean = byte = c16 = c8 = char = complex128 = None
    complex64 = double = f4 = f8 = ffi = ffi_forced_object = None
    float32 = float64 = float_ = i1 = i2 = i4 = i8 = int16 = None
    int32 = int64 = int8 = int_ = intc = intp = long_ = longlong = None
    none = short = u1 = u2 = u4 = u8 = uchar = uint = None
    uint16 = uint32 = uint64 = uint8 = uintc = uintp = ulong = None
    ulonglong = ushort = void = None

    prange = range

if sc:
    del sc

    from scipy.optimize import newton
    from scipy.sparse import diags ### construction des matrices ###

if cp:
    del cp

    import cvxpy as cp

sys.setrecursionlimit(10**8)

## Mathematics tools

real = TypeVar('real', int, float)
number = TypeVar('number', real, complex)
vec = TypeVar('vec', number, np.ndarray)

sqrt: Callable[[vec], vec] = sq

τ: float = 2*π


def d(f, order:int):
    """Approximate the derivative of a function.
        
    Keyword arguments:
        f -- the function to be derived
        order -- order of the derivative
        
    """
    order = int(order)
    if order < 0: 
        raise TypeError('Negative arguments are not supported yet.')
    δ = d(f, order - 1)
    return (lambda a: (δ(a+1e-300)-δ(a))/1e-300) if order else f

class MathFunc:
    """Create a mathematical function easier to manipulate than built-in 
    functions.
    
    """
    def __abs__(self:'MathFunc')->'MathFunc':
        """Return absolute value of the function."""
        return MathFunc(
            f=eval(
            'lambda ' + ','.join(self.var) + ':abs(self('
            + ','.join(self.var) + '))',{'self':self} )
            )
    
    def __add__(self:'MathFunc', value)->'MathFunc':
        """Return function added with the value"""
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':self('
            + ','.join(self.var) + ')+(v( ' + ','.join(self.var)
            + ')if callable(v) else v)', {'v':value, 'self':self} )
            )
    
    def __and__(self:'MathFunc', value)->'MathFunc':
        return 'convolution'
    
    def __bool__(self:'MathFunc')->bool:
        """Check if the function is probably zero."""
        return any({
            self(*[π*(random()-.5) for v in self.var]) 
            for r in range(100)
            })
    
    def __call__(self:'MathFunc', *args):
        """Call the f function. self(*args)"""
        if(len(args) < self.f.__code__.co_argcount):
            args += (0,)*(self.f.__code__.co_argcount-len(args))
        args = args[:self.f.__code__.co_argcount]
        return self.f(*args)
    
    def __divmod__(self:'MathFunc', value)->'MathFunc':
        """Division euclidienne de f || divmod(f)"""
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':divmod(self('
            + ','.join(self.var) + '),v( ' + ','.join(self.var)
            + ')if(callable(v)) else v)', {'v':value, 'self':self} ))
    
    def __eq__(self:'MathFunc', value)->bool:
        """Check if function is probably  || self == value"""
        return bool(self - value)
    
    def __floordiv__(self:'MathFunc', value)->'MathFunc':
        """Return floor division of the function by the value"""
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':self('
            + ','.join(self.var) + ')//(v( ' + ','.join(self.var)
            + ')if callable(v) else v)', {'v':value, 'self':self} )
            )
    
    def __init__(self:'MathFunc', var: str = 'x', eq: str ='x', f=None)->None:
        if f is None:
            self.var=var.split(',')
            exec('self.f=lambda ' + var + ':' + eq)
        else:
            self.f, self.var = f, list(f.__code__.co_varnames)
            
    def __mul__(self:'MathFunc', value)->'MathFunc':
    ##Multiplication à droite || self*value
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':self('
            + ','.join(self.var) + ')*(v( ' + ','.join(self.var)
            + ')if callable(v) else v)', {'v':value, 'self':self} )
            )
    
    def __neg__(self:'MathFunc')->'MathFunc':##Opposé de f || -f
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':-self('
            + ','.join(self.var) + ')', {'self':self} )
            )
    
    def __pos__(self:'MathFunc')->'MathFunc':## ||+f
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ': +self('
            + ','.join(self.var) + ')', {'self':self} )
            )
    
    def __radd__(self:'MathFunc', value)->'MathFunc':
    ##Addition à gauche || value+self
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':(v( '
            + ','.join(self.var) + ')if callable(v) else v) + self('
            + ','.join(self.var) + ')', {'v':value, 'self':self} )
            )
    
    def __rmul__(self:'MathFunc', value)->'MathFunc':
    ##Multiplication à gauche || value*self
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':(v( '
            + ','.join(self.var) + ')if callable(v) else v) * self('
            + ','.join(self.var) + ')', {'v':value, 'self':self} )
            )
    
    def __rsub__(self:'MathFunc', value)->'MathFunc':
    ##Soustraction par f || value-self
        return MathFunc(
            f=eval('lambda ' + ','.join(self.var) + ':(v( '
            + ','.join(self.var) + ')if callable(v) else v) - self('
            + ','.join(self.var) + ')', {'v':value, 'self':self} )
            )
    
    def __sub__(self:'MathFunc', value)->'MathFunc':
    ##Soustraction de f || self-value
        return self+(-value)
    
    def noname(self:'MathFunc', order=1)->'MathFunc':
        return MathFunc(f=d(self,order))

@njit(fastmath=True)
def njit_norm(X:np.ndarray, k:real = χ)->real:
    return norm(X, k)

@njit(fastmath=True, parallel=True)
def njit_linspace(start:real, stop:real, step:int):
    return np.linspace(start, stop, step)

class Cauchy:
    """ Compute Euler, backward Euler and Runge-Kutta methods to resolve 
        an initial value problem (IVP or Cauchy problem):
             dy/dt = F(t, y), y(0) = y0 where f and y0 are given

        Initialization:
        T -- maximum time of the solution
        y0 -- initial value
        F -- function of problem
        N -- number of time subdivisions
        q, A, B, C -- parameters of Runge-Kutta method
        """
    def __init__(self:'Cauchy', **kwargs) -> None:
        if ('numpy' not in sys.modules 
                or 'matplotlib' not in sys.modules):
            print(
                "numpy or matplotlib modules not found."
                + " Please retry after (re)install."
                )
            sys.exit(1)
        self.T: int = 1
        self.y0: vec = 0
        def F(t:float, y:vec)->vec: 
            return self.y0
        self.F: Callable[[float, vec], vec] = F
        self.a: int = 1
        self.N: int = 1000
        self.q: Optional[Union[int,str]] = '2'
        self.X: Optional[np.ndarray] = ar([0])
        self.Yl: Optional[np.ndarray] = None
        self.A: np.ndarray = ar([
            [0, 0], 
            [.5, 0]
            ])
        self.B: np.ndarray = ar(
            [0,1]
            )
        self.C: np.ndarray = ar(
            [0,.5]
            )
        self.eul_exp: Callable[['Cauchy'], None] = self.euler_explicit
        self.eul_imp: Callable[['Cauchy'], None] = self.euler_implicit
        self.run_kut: Callable[['Cauchy'], None] = self.runge_kutta
        self.run_kut_2: Callable[['Cauchy'], None] = self.runge_kutta_2
        self.run_kut_4: Callable[['Cauchy'], None] = self.runge_kutta_4
        self.update(**kwargs)
        
    def update(self:'Cauchy', **kwargs) -> None:
        for k in kwargs: exec("self."+k+"=kwargs[k]")

        self.Yexp: Union[list, np.ndarray] = [self.y0] 
        self.Yimp: Union[list, np.ndarray] = [self.y0]
        self.Yr_k: Union[list, np.ndarray] = [self.y0]

        if len(self.X) != self.N+2 or self.X[-1] != self.T: 
            self.X: np.ndarray = ar(
                [k*self.T/(self.N+1) for k in range(self.N+2)]
                )
        if (self.q in {'RK2', '2'} 
                or (self.q==2 and 2 not in self.A.shape)):
            self.A: np.ndarray = ar([
                [0, 0], 
                [.5, 0]
                ])
            self.B: np.ndarray = ar(
                [0, 1]
                )
            self.C: np.ndarray = ar(
                [0, .5]
                )
            self.q: Optional[Union[int,str]] = 2
        elif (self.q in {'RK4', '4'} 
                or (self.q==4 and 4 not in self.A.shape)):
            self.A: np.ndarray = ar([
                [0, 0, 0, 0], 
                [.5, 0, 0, 0],
                [0, .5, 0, 0], 
                [0, 0, 1, 0]
                ])
            self.B: np.ndarray = ar(
                [1/6, 1/3, 1/3, 1/6]
                )
            self.C: np.ndarray = ar(
                [0, .5, .5, 1]
                )
            self.q: Optional[Union[int,str]] = 4

    def euler_explicit(self:'Cauchy') -> None:
        """Compute Euler explicit method (forward Euler method)  """
        while (lambda N=len(self.Yexp), X=self.X, Y=self.Yexp, F=self.F: 
            Y.append(Y[-1]+F(X[N], Y[-1])*X[N]/N)
            or not Y[self.N+1:])(): ...
        self.Yexp: np.ndarray = ar(self.Yexp)
        self.Yl: np.ndarray = self.Yexp

    def euler_implicit(self:'Cauchy') -> None:
        while (lambda N=len(self.Yimp), X=self.X, Y=self.Yimp, F=self.F:
            Y.append(newton((lambda u:u-Y[-1]-F(X[N],u)*X[N]/N),Y[-1]))
            or not Y[self.N+1:])(): ... 
        self.Yimp: np.ndarray = ar(self.Yimp)
        self.Yl: np.ndarray = self.Yimp 
        
    def runge_kutta(self:'Cauchy') -> None:
        def f(N=len(self.Yr_k), X=self.X, Y=self.Yr_k, F=self.F, 
            A=self.A, B=self.B, C=self.C, q=self.q):
            δ: float = X[N]/N
            l: List[float] = []
            for r in range(q): 
                l.append((F(X[N] + C[r]*δ, Y[-1]+A[r][:r]@l*δ)))
            Y.append(B@l*δ + Y[-1]) 
            return not Y[self.N+1:]

        while f(): ...
        self.Yr_k: np.ndarray = ar(self.Yr_k)
        self.Yl: np.ndarray = self.Yr_k

    def runge_kutta_2(self:'Cauchy') -> None:
        self.q = 'RK2'
        self.update()
        return self.runge_kutta()

    def runge_kutta_4(self:'Cauchy') -> None:
        self.q = 'RK4'
        self.update()
        return self.runge_kutta()

    def errmax(
            self:'Cauchy', y:np.ndarray=None, yref: np.ndarray = None
            )->real:
        return njit_norm(
            (y if y is not None else self.Yl) 
            - (yref if yref is not None else self.Y), 
            χ
            )

    def display_2d(
            self:'Cauchy', alg: str = 'runge_kutta', 
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        self.dp_2d: Display2d = Display2d(func, cauchy=self, **kwargs)

    def display_2d_animated(
            self:'Cauchy', alg: str = 'runge_kutta', 
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        self.dp_2d_a: Display2dAnimated = Display2dAnimated(
            func, cauchy=self, **kwargs
            )

    def display_3d(
            self:'Cauchy', alg: str = 'runge_kutta',
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        self.dp_3d: Display3d = Display3d(
            func, cauchy=self, **kwargs
            )

    def display_3d_animated(
            self:'Cauchy', alg: str = 'runge_kutta', 
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        self.dp_3d_a: Display3dAnimated = Display3dAnimated(
            func, cauchy=self, **kwargs
            )

class Transport:
    def __init__(self:'Transport', **kwargs) -> None:
        def f(t: real = 0, x: real = 0) -> vec: return 0
        def u0(x: vec): return 0*x
        self.f: Callable[[real, real], vec] = f
        self.u0: Callable[[real], vec] = u0
        self.M: int = 1000
        self.N: int = self.M
        self.T: real = 1
        self.Ul: Optional[np.ndarray] = None
        self.update(**kwargs)

    def update(self:'Transport', **kwargs) -> None:
        for k in kwargs: exec("self."+k+"=kwargs[k]")
        if 'M' in kwargs and 'N' not in kwargs: self.N = self.M 
        if 'N' in kwargs and 'M' not in kwargs: self.M = self.N 
        self.δx: float = 1/(self.M+1)
        self.δt: float = 1/(self.N+1)
        self.T0 = njit_linspace(0, self.T, self.M+2)
        self.u0x: np.ndarray = self.u0(self.T0)

@njit(parallel=True, fastmath=True)
def scalar_forward_time_backward_space(
        u0x: np.ndarray, c:real, M:np.ndarray, N: np.ndarray, δx:real, 
        δt:real
        ) -> None:
    λ: float = c*δt/δx
    γ_1: float = λ
    γ0: float = 1 - λ
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = γ_1*U[n][i-1] + γ0*U[n][i]
    return U

@njit(parallel=True, fastmath=True)
def scalar_forward_time_forward_space(
        u0x: np.ndarray, c:real, M:np.ndarray, N: np.ndarray, δx:real, 
        δt:real
        ) -> None:
    λ: float = c*δt/δx
    γ0: float = 1 + λ
    γ1: float = -λ
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 2, 0):
            U[n+1][i] = γ0*U[n][i] + γ1*U[n][i+1]
    return U

@njit(parallel=True, fastmath=True)
def scalar_forward_time_centered_space(
        u0x: np.ndarray, c:real, M:np.ndarray, N: np.ndarray, δx:real, 
        δt:real
        ) -> None:
    λ: float = c*δt/δx
    γ_1: float = λ/2
    γ0: float = 1 
    γ1: float = -λ/2
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = γ_1*U[n][i-1] + γ0*U[n][i] + γ1*U[n][i+1]
    return U

@njit(parallel=True, fastmath=True)
def scalar_lax_wendroff(
        u0x: np.ndarray, c:real, M:np.ndarray, N: np.ndarray, δx:real, 
        δt:real
        ) -> None:
    λ: float = c*δt/δx
    γ_1: float = λ*(1+λ)/2
    γ0: float = 1 - λ**2
    γ1: float = λ*(λ-1)/2
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = γ_1*U[n][i-1] + γ0*U[n][i] + γ1*U[n][i+1]
    return U

class ScalarTransport(Transport):
    """docstring for ScalarTransport"""
    def __init__(self, **kwargs) -> None:
        self.c: real = 1/2
        #def c(t: real = 0, x: real = 0) -> vec:
        #    return 1
        #self.c: Callable[[real, real], vec] = c
        super(ScalarTransport, self).__init__(**kwargs)
        self.f_t_f_s: Callable[['ScalarTransport'], None] = \
            self.forward_time_forward_space
        self.f_t_b_s: Callable[['ScalarTransport'], None] = \
            self.forward_time_backward_space
        self.f_t_c_s: Callable[['ScalarTransport'], None] = \
            self.forward_time_centered_space

    def forward_time_forward_space(self:'ScalarTransport') -> None:
        self.Uftfs: np.ndarray = scalar_forward_time_forward_space(
            self.u0x, self.c, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftfs

    def forward_time_backward_space(self:'ScalarTransport') -> None:
        self.Uftbs: np.ndarray = scalar_forward_time_backward_space(
            self.u0x, self.c, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftbs

    def forward_time_centered_space(self:'ScalarTransport') -> None:
        self.Uftcs: np.ndarray = scalar_forward_time_centered_space(
            self.u0x, self.c, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftcs

    def lax_wendroff(self:'ScalarTransport') -> None:
        self.Ul_w: np.ndarray = scalar_lax_wendroff(
            self.u0x, self.c, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Ul_w

    """def lax_friedrichs(self:'ScalarTransport') -> None:
        self.Ul = self.Ul_f = scalar_lax_friedrichs(
            self.u0x, self.c, self.M, self.N, self.δx, self.δt 
            )"""
    def display_2d_animated(
            self:'ScalarTransport', alg: str = 'lax_wendroff', 
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        def transport_init(d:Display2dAnimated) -> Callable[[],List]:
            def init():
                d.lines, = d.ax.plot(self.T0, self.Ul[0])
                return d.lines,
            return init

        def transport_data_generator(
                d:Display2dAnimated
                ) -> Callable[[],List]:
            def d_g():
                for i in (*range(d.frames), -1):
                    l = round(i*self.N/d.frames)
                    yield self.Ul[l]
            return d_g

        def transport_update(
                d:Display2dAnimated
                )-> Callable[[List],List]:
            def upd(data):
                d.lines.set_data(self.T0, data)
                return d.lines,
            return upd

        self.dp_2d_a: Display2dAnimated = Display2dAnimated(
            func=func, cauchy=self, init=transport_init, 
            data_generator=transport_data_generator, 
            update=transport_update, ignore_X=True, **kwargs
            )

@njit(parallel=True, fastmath=True)
def vector_forward_time_backward_space(
        u0x: np.ndarray, A:np.ndarray, M:np.ndarray, N: np.ndarray,
        δx:real, δt:real
        ) -> None:
    λ: np.ndarray = A*δt/δx
    d: int  = u0x[0].shape[0]
    I: np.array = np.identity(d)
    Γ_1: np.ndarray = λ
    Γ0: np.ndarray = I - λ
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = Γ_1@U[n][i-1] + Γ0@U[n][i]
    return U

@njit(parallel=True, fastmath=True)
def vector_forward_time_forward_space(
        u0x: np.ndarray, A:np.ndarray, M:np.ndarray, N: np.ndarray, 
        δx:real, δt:real
        ) -> None:
    λ: np.ndarray = A*δt/δx
    d: int  = u0x[0].shape[0]
    I: np.array = np.identity(d)
    Γ0: np.ndarray = I + λ
    Γ1: np.ndarray = -λ 
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 2, 0):
            U[n+1][i] = Γ0@U[n][i] + Γ1@U[n][i+1]
    return U

@njit(parallel=True, fastmath=True)
def vector_forward_time_centered_space(
        u0x: np.ndarray, A:np.ndarray, M:np.ndarray, N: np.ndarray, 
        δx:real, δt:real
        ) -> None:
    λ: np.ndarray = A*δt/δx
    d: int  = u0x[0].shape[0]
    I: np.array = np.identity(d)
    Γ_1: float = λ/2
    Γ0: float = I 
    Γ1: float = -λ/2
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = Γ_1@U[n][i-1] + Γ0@U[n][i] + Γ1@U[n][i+1]
    return U

@njit(parallel=True, fastmath=True)
def vector_lax_wendroff(
        u0x: np.ndarray, A:np.ndarray, M:np.ndarray, N: np.ndarray,
        δx:real, δt:real
        ) -> None:
    λ: np.ndarray = A*δt/δx
    d: int  = u0x[0].shape[0]
    I: np.array = np.identity(d)
    Γ_1: float = λ@(I+λ)/2
    Γ0: float = 1 - λ**2
    Γ1: float = λ@(λ-I)/2
    U: np.ndarray = np.empty((N+2, M+2), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            U[n+1][i] = Γ_1*U[n][i-1] + Γ0*U[n][i] + Γ1*U[n][i+1]
    return U

@njit(fastmath=True, parallel=True)
def vector_lax_friedrichs(
        u0x: np.ndarray, A:np.ndarray, M:np.ndarray, N: np.ndarray, 
        δx:real, δt:real
        ) -> None:
    λ: float = δx/δt
    d: int  = u0x[0].shape[0]
    U: np.ndarray = np.empty((N+2, M+2)+(d,), dtype='float64')
    U[0]: np.ndarray = u0x
    for n in range(N + 1):
        for i in prange(-M - 1, 1):
            α = (U[n][i-1] + U[n][i+1])/2
            β = (U[n][i-1] - U[n][i+1])*λ/2
            U[n+1][i] = α + A@β
    return U

class VectorTransport(Transport):
    """docstring for VectorTransport"""
    def __init__(self:'VectorTransport', **kwargs) -> None:
        super(VectorTransport, self).__init__(**kwargs)
        self.f_t_f_s: Callable[['VectorTransport'], None] = \
            self.forward_time_forward_space
        self.f_t_b_s: Callable[['VectorTransport'], None] = \
            self.forward_time_backward_space
        self.f_t_c_s: Callable[['VectorTransport'], None] = \
            self.forward_time_centered_space

    def forward_time_forward_space(self:'VectorTransport') -> None:
        self.Uftfs: np.ndarray = vector_forward_time_forward_space(
            self.u0x, self.A, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftfs

    def forward_time_backward_space(self:'VectorTransport') -> None:
        self.Uftbs = vector_forward_time_backward_space(
            self.u0x, self.A, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftbs

    def forward_time_centered_space(self:'VectorTransport') -> None:
        self.Uftcs: np.ndarray = vector_forward_time_centered_space(
            self.u0x, self.A, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Uftcs

    def lax_wendroff(self:'VectorTransport') -> None:
        self.Ul_w: np.ndarray = vector_lax_wendroff(
            self.u0x, self.A, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Ul_w

    def lax_friedrichs(self:'VectorTransport') -> None:
        self.Ul_f: np.ndarray = vector_lax_friedrichs(
            self.u0x, self.A, self.M, self.N, self.δx, self.δt 
            )
        self.Ul: np.ndarray = self.Ul_f 

    def display_2d_animated(
            self:'VectorTransport', alg: str = 'lax_friedrichs', 
            func: Optional[Callable] = None, **kwargs
            ) -> None:
        if alg: eval('self.' + alg)()
        def transport_init(d:Display2dAnimated):
            def init():
                d.lines = d.ax.plot(self.T0, self.Ul[0])
                return d.lines
            return init

        def transport_data_generator(d:Display2dAnimated):
            def d_g():
                for i in (*range(d.frames), -1):
                    l = round(i*self.N/d.frames)
                    yield self.Ul[l].T
            return d_g

        def transport_update(d:Display2dAnimated):
            def upd(data):
                for i in range(len(d.lines)):
                    d.lines[i].set_data(self.T0, data[i])
                return d.lines
            return upd

        self.dp_2d_a: Display2dAnimated = Display2dAnimated(
            func=func, cauchy=self, init=transport_init, 
            data_generator=transport_data_generator, 
            update=transport_update, ignore_X=True, **kwargs
            )


@njit(fastmath=True, parallel=True)
def complex_function_compute(x, y, δ, f):
    print(len(x),len(y))
    F = _0((len(x), len(y)),complex128)
    F = f(x + y*1j)
    ρ = np.abs(F)
    return F, ρ
        
def isn(t): return t!=t
class ComplexFunction:
    def __init__(self:'ComplexFunction', *args, **kwargs):
        self.rcount: int = 50
        self.ccount: int  = 50
        self.δ: real  = .01

        @njit(parallel=True)
        def f(z:vec) -> vec:
            return 1/(1+z**2)
        self.f: Callable[[vec], vec] = kwargs.get('f',f)
        del f
        self.z1: number = kwargs.get('z1', -2 - 2j)
        self.z2: number = kwargs.get('z2', 2 + 2j)
        self.zmin: number = 0
        self.zmax: number = 4
        self.update(**kwargs)

    def update(self:'ComplexFunction', *args, **kwargs):
        for k in kwargs: exec("self."+k+"=kwargs[k]")
        self.xmin: real = min((self.z1.real, self.z2.real))
        self.xmax: real = self.z1.real + self.z2.real - self.xmin
        self.ymin: real = min((self.z1.imag, self.z2.imag))
        self.ymax: real = self.z1.imag + self.z2.imag - self.ymin

    def calcul(self:'ComplexFunction', **kwargs):
        self.x, self.y = np.meshgrid(
            np.arange(self.xmin/self.δ, self.xmax/self.δ)*self.δ, 
            np.arange(self.ymin/self.δ, self.ymax/self.δ)*self.δ
            )
        self.F, self.ρ = complex_function_compute(
            self.x, self.y, self.δ, self.f
            )
        self.θ = np.array([
            [cmath.phase(fz) if(abs(fz) != np.inf) else 0 for fz in _]
            for _ in self.F
            ])


    def singularités(self:'ComplexFunction', **kwargs):
        for y in range(len(self.ρ)):
            for x in range(len(self.ρ[y])):
                z = (x+1j*y)*self.δ+self.xmin+self.ymin*1j
                bl = (self.ρ[y][x] == np.inf or isn(self.θ[y][x]) 
                    or isn(self.ρ[y][x])
                    )
                if not z:
                    print(
                        'Valeur en zéro:', self.ρ[y][x], self.θ[y][x], 
                        end=' '+'\n'*(1-bl)
                        )
                if bl:
                    ε = sum(
                        self.f(ar([z + 1j**_*10**-12 for _ in range(4)])
                        )/4
                        )
                    print(
                        'Point critique en', z, 'ρ=', self.ρ[y][x], 
                        ' θ=', self.θ[y][x], ' valeurs approchée: ρ=', 
                        abs(ε), ' θ=', cmath.phase(ε)
                        )
                    if isn(self.ρ[y][x]): self.ρ[y][x] = abs(ε)
                    if isn(self.θ[y][x]): self.θ[y][x] = cmath.phase(ε)
                elif (not self.ρ[y][x]): 
                    print('Zéro en', z, ' θ= ', self.θ[y][x])
                    
    def graphique(self:'ComplexFunction', **kwargs):
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.surf = self.ax.plot_surface(
            self.x, self.y, self.ρ, alpha=1, antialiased=False,
            rcount=self.rcount, ccount=self.ccount, cmap=cm.jet, 
            facecolors=cm.jet(colors.Normalize(-π,π)(self.θ))
            )
        self.ax.set_zlim(self.zmin, self.zmax)
        #for _ in ['x','y']: 
        #    self.ax.callbacks.connect(_+'lim_changed', self.update(_))
        self.ax.set_autoscale_on(False)
        plt.show()

    def ud(self:'ComplexFunction', **kwargs):
        print(self.ax.viewLim.intervalx) 
        self.xmin, self.xmax = self.ax.viewLim.intervalx
        print(self.ax.viewLim.intervaly)
        self.ymin, self.ymax = self.ax.viewLim.intervaly
        print(self.xmin)
        self.calcul(self.δ)
        self.ax.figure.canvas.draw_idle()

    def nettoyage(self:'ComplexFunction'):
        self.ρ = 0
        self.θ = 0

    def __call__(self:'ComplexFunction', **kwargs):
        self.calcul(**kwargs)
        if kwargs.get('singularités'):
            self.singularités(**kwargs)
        if kwargs.get('graphique', True):
            self.graphique(**kwargs)

def V(t, f=(lambda x:x)):
    return [
        f(x) if ( callable(f)) else f 
            for x in (range(t)if type(t) is int else t)
        ]#f(ar(range(t)))+(_ and 0*V(t,_=0))

class HeatEquation:
    def __init__(self:'HeatEquation', **kwargs):
        self.l, self.n, self.f, self.ε = 1, 20, lambda x,y=0: 1, .01
        self.ω, self.u0, self.un, self.uext = 1.5, 0, 0, 0
        self.alg, self.x0, self.c = default_alg, None, None
        self.update(**kwargs)

    def update(self:'HeatEquation', **kwargs):
        for k in kwargs: exec("self."+k+" = kwargs[k]")
        self.Δx = self.l/self.n
        self.x0 = np.ones(self.n+1) if self.x0 is None else self.x0
    
    def ub(self:'HeatEquation'): 
        return (
            (
                self.unid(), 
                plt.plot(njit_linspace(0, 1, self.n+1), self.unidmat[0])
                ) if (self.c or self.f.__code__.co_argcount) == 1
            else (
                self.bid(), 
                plt.figure().gca(projection='3d').plot_surface(
                    *mg(*(njit_linspace(0, l, self.n+1),)*2), 
                    self.bidmat[0]
                    )
                )
            )#interface graphique

    def unid(self:'HeatEquation'):
        self.unidmat = self.alg(
            ar([[self.Δx**2] + self.n*[0]]
                + [
                    [
                        (abs(j+1-i) == 1) - 2*(i==j+1) 
                        for i in range(self.n+1)
                        ] 
                    for j in range(self.n-1)
                    ]
                + [self.n*[0] + [self.Δx**2]]
                ),
            (
                V( 
                    self.n+1, lambda x:-self.f(x*self.Δx)
                    )[1:-1]@np.eye(self.n-1,self.n+1,1)
                - ar([self.u0] + (self.n-1)*[0] + [self.un])
                ) * self.Δx**2, 
            x0=self.x0, ε=self.ε, ω=self.ω
            )#calcul et retour

    def bid(self:'HeatEquation'):
        z1 = self.alg(
            [
                [
                    4*(i==j) or -(max(i,j)%(self.n-1) 
                    and ((i-j)**2 == 1)) or -(abs(i-j)==self.n-1) 
                    for i in range((self.n-1)**2)
                    ]
                for j in range((self.n-1)**2)
                ],
            [
                self.f(
                    (1 + x%(self.n-1))*self.l/self.n,
                    (1 + x//(self.n-1))*self.l/self.n
                    ) 
                * (self.l/self.n)**2 for x in range((self.n-1)**2)
                ], 
            x0, ε=self.ε, ω=self.ω
            )#dépliage et calcul
        self.bidmat = ar(
            [
                [
                    i and self.n-i and j and self.n-j 
                    and z1[0][i-1 + (j-1)*(self.n-1)] 
                    for i in range(self.n+1)
                    ] 
                for j in range(self.n+1)
                ]
            ),z1[1]#repliage et retour

@njit(fastmath=True, parallel=True)
def julia_color(img, c, dX, dY, X, Y, R, count, bw):
    for y in prange(Y):
        for x in prange(X):
            z: float64 = 2*(dX*(x/X-.5) + dY*(.5-y/Y)*1j)
            for i in range(count):
                if abs(z) > R:
                    z = 2
                    break
                z = z**2 + c
            if bw:
                img[y,x] = (255,)*3 if i == count-1 else (0,)*3
            else:
                ρ:int = int(abs(z)*(1<<24))
                r:int = ρ >> 16
                g:int = (ρ - (r<<16)) >> 8
                b:int = ρ - (r<<16) - (g<<8)
                img[y,x] = (r,g,b)
    return img

class Julia:
    def __init__(self:'Julia', c:complex = None, dX:real = None,
                X:int=None, *args, **kwargs):
        self.dX: float = float(dX if dX is not None else 1.75)
        self.dY: float = float(kwargs.get('dY', dX))
        self.X: int = int(X) if X is not None else 2000
        self.Y: int = int(kwargs.get('Y', self.dY/self.dX * self.X))
        self.image = np.empty((self.Y,self.X)+(3,), dtype=np.uint8)
        self.count: int = kwargs.get('count', 255)
        self.c: number = complex(
            c if c is not None else complex(-0.8, 0.156)
            )
        self.bw: bool = kwargs.get('bw', False)
        self.R:int = 0
        while self.R**2-self.R < (self.dX+self.dY)*sqrt(2*self.c)/2:
            self.R += 1


    def show(self:'Julia'): 
        plt.imshow(self.image, extent=[
            -self.dX, self.dX, -self.dY, self.dY
            ])
        plt.show()

    def color(self:'Julia', *args, **kwargs):
        self.image = julia_color(
            self.image, self.c, self.dX, self.dY, self.X, self.Y, self.R,
            self.count, self.bw
            )
        
    def save(self:'Julia'): 
        mpimg.imsave(
            input("Nom du fichier de sauvegarde: ") + ".png", self.image
            )

    def __call__(self:'Julia', *args, **kwargs):
        self.color(*args, **kwargs)
        self.show()

##Calcul de solutions matricielles de type Ax = b
def γ(M, _1=False): return dg((lambda x:1/x if(_1)else x)(dg(M)))
class MatrixInversion:
    def __init__(self:'MatrixInversion', A, b, x0, ε=.01, n=χ, ω=0):
        self.A: np.ndarray = A
        self.b: np.ndarray = b
        self.x0: np.ndarray = x0
        self.ε: real = ε
        self.n: int = n
        self.ω: real = ω

    def default_alg(self:'MatrixInversion'): return S(self.A, self.b), 0

    def jacobi(self:'MatrixInversion'):
        D_, _E_F = γ(self.A, True), U(self.A, 1) + L(self.A, -1)
        def φ(x, x_1=χ, _=0):
            return ((x, _) if norm(x - x_1) < ε or _ > n 
                else φ(D_ @ (self.b-_E_F@x), x, _+1))
        return φ(x0)

    def jacobi_iter(self:'MatrixInversion'):
        D_, _E_F, x = γ(self.A, True), U(self.A, 1) + L(self.A, -1), x0
        for _ in range(n if n != χ else 25): x = D_@(b - _E_F@x)
        return x, n

    def gauss_seidel(self:'MatrixInversion'):
        _1lA = _1(L(self.A)); B_gs, c = _1lA@-U(self.A, 1), _1lA@b
        def φ(x, x_1=χ, _=0):
            return ((x, _) if(norm(x - x_1) < ε or _ > n) 
                else φ(B_gs@x + c, x, _+1))
        return φ(x0)

    def sor(self:'MatrixInversion'):
        T = _1(L(self.A, -1) + γ(self.A)/ω) 
        B_ω, c = T@((1-ω)/ω*γ(self.A)-U(self.A, 1)), T@b 
        def φ(x, x_1=χ, _=0): 
            return ((x,_) if(norm(x - x_1) < ε or _ > n) 
                else φ(B_ω@x + c, x, _+1))
        return φ(x0)

    def cholesky(self:'MatrixInversion'):
        L, n = _0((self.A.shape[0],)*2), self.A.shape[0]
        for i in range(n):
            for j in range(i+1): 
                L.itemset(
                    (i, j), 
                    (
                        self.A[i, j] 
                        - sum({L[i, k] * L[j, k] for k in range(j)})
                        ) / L[j, j] if i>j
                    else sq(
                        self.A[i, j] 
                        - sum({L[i, k]**2 for k in range(i)})
                        )
                    )
        return L

class Display:
    def get_data_from_cauchy(self:'Display', anim=False, **kwargs):
        self.N: int = self.cauchy.N
        if anim:
            self.X: np.ndarray = kwargs.get('X', 
                (
                    (
                        self.cauchy.Yl.T
                        if kwargs.get('keep_complex')
                        else self.cauchy.Yl.real.T 
                        ) 
                    if len(self.cauchy.Yl) > 1 
                    else self.cauchy.Yl
                    )
                )

            try:
                if self.X == [self.y0]:
                    warn('Cauchy class provided without data')
            except AttributeError:
                ...
        else:
            self.X: np.ndarray = np.stack(
                (self.cauchy.X, self.cauchy.Yl)
                )

    def get_data_from_input(self:'Display', anim=False, **kwargs):
        try:
            self.X: np.ndarray = (
                kwargs['X'] if kwargs.get('keep_complex')
                else kwargs['X'].real
                )
            self.N: int = len(self.X[0])
        except AttributeError:
            self.X: np.ndarray = kwargs['X']
            self.N: int = len(self.X[0])
        except KeyError:
            if not kwargs.get('ignore_X'):
                warn("Display missing 'X' argument")


    def __init__(self:'Display', **kwargs):
        self.__dict__.update(kwargs)

        self.fig: Optional[matplotlib.figure.Figure] = kwargs.get(
            'figure', kwargs.get('fig')
            )
        if not self.fig:
            self.fig: matplotlib.figure.Figure = plt.figure(
                **kwargs.get('plt_figure_args', {})
                )


    def __init__2d__(self:'Display', **kwargs):
        self.ax = self.fig.gca()

        self.ax.set_title(kwargs.get('title'))
        self.ax.set_xlabel(kwargs.get('xlabel','X'))
        self.ax.set_ylabel(kwargs.get('ylabel','Y'))
        try:
            self.ax.set_xlim(kwargs['xlim'])
            self.ax.set_ylim(kwargs['ylim'])
        except KeyError:
            ...

    def __init__3d__(self:'Display', **kwargs):
        self.ax = self.fig.gca(projection='3d')

        self.__init__2d__(**kwargs)
        self.ax.set_zlabel(kwargs.get('zlabel','Z'))
        #self.ax.set_zlim(kwargs.get('zlim'))


    def __init__animation__(self:'Display', func=None, **kwargs):
        self.frames: int = kwargs.get('frames', 1000)
        trid = type(self) == Display3dAnimated
        try: self.get_data_from_cauchy(anim=True, **kwargs)
        except AttributeError: 
            self.get_data_from_input(anim=True, **kwargs)

        def default_init():
            try:
                self.__dict__.get('lines').remove()
            except AttributeError:
                ...
            self.lines, = self.ax.plot(
                *self.X, **kwargs.get('ax_plot_args', {})
                )
            return self.lines,

        try:
            self.init = kwargs['init'](self)
        except KeyError:
            self.init = default_init

        def default_data_generator():
            for i in range(self.frames + 1):
                yield self.X[:,:round(i*self.N/self.frames)]

        try:
            self.data_generator = kwargs[
                'data_generator'
                ](self)
        except KeyError:
            self.data_generator: Callable\
                = default_data_generator
                
        def default_update(data): 
            self.lines.set_data(data)
            return self.lines,
        
        def default_update_3d(data): 
            self.lines.set_data_3d(data)
            return self.lines,

        try:
            self.update = kwargs['update'](self)
        except KeyError:
            self.update = default_update_3d if trid else default_update

        if func: func(**kwargs.get('func_args', {}))

        self.ani = ani.FuncAnimation(
            init_func=self.init, fig=self.fig, func=self.update,
            frames=self.data_generator, 
            **kwargs.get('ani_args',{'interval':1})
            )

        if kwargs.get('plt_show'): plt.show()


    def __init__static__(self:'Display', func=None, **kwargs): 
        try: self.get_data_from_cauchy(anim=False, **kwargs)
        except AttributeError: self.get_data_from_input(False, **kwargs)

        def default_init():
            self.lines, = self.ax.plot(
                *self.X, **kwargs.get('ax_plot_args', {})
                )
            return self.lines,

        try:
            self.init = kwargs['init'](self)
        except KeyError:
            self.init = default_init

        if func: func(**kwargs.get('func_args', {}))
        if kwargs.get('plt_show'): plt.show()

class Display2d(Display):
    """docstring for Display2d"""
    def __init__(self:'Display2d', func=None,**kwargs):
        super(Display2d, self).__init__(**kwargs)
        super(Display2d, self).__init__2d__(**kwargs)
        super(Display2d, self).__init__static__(func, **kwargs)

class Display2dAnimated(Display):
    """docstring for Display2dAnimated"""
    def __init__(self:'Display2dAnimated', func=None, **kwargs):
        super(Display2dAnimated, self).__init__(**kwargs)
        super(Display2dAnimated, self).__init__2d__(**kwargs)
        super(Display2dAnimated, self).__init__animation__(
            func, **kwargs
            )

class Display3d(Display):
    """docstring for Display3d"""
    def __init__(self:'Display3d', func=None, **kwargs):
        super(Display3d, self).__init__(**kwargs)        
        super(Display3d, self).__init__3d__(**kwargs)        
        super(Display3d, self).__init__static__(func, **kwargs)        

class Display3dAnimated(Display):
    """docstring for Display3dAnimated"""
    def __init__(self:'Display3dAnimated', func=None, **kwargs):
        super(Display3dAnimated, self).__init__(**kwargs)
        super(Display3dAnimated, self).__init__3d__(**kwargs)
        super(Display3dAnimated, self).__init__animation__(
            func, **kwargs
            )

class Koch:
    def __init__(self:'Koch', *args, **kwargs):
        self.φ = (sq(5)+1)/2
        self.height: int = 800
        self.width: int = 1200
        self.n: int = 5
        self.d: real = None
        self.angle: real = π/3
        self.inverse: bool = False
        self.α = π/2
        self.update(**kwargs)

        # réglages de la tortue
        speed(0)  # vitesse (0 = vitesse max)
        hideturtle()  # cacher la tortue
        radians()

        # création de la fenêtre
        setup(self.width, self.height)

    def update(self:'Koch', **kwargs):
        for k in kwargs: exec("self."+k+" = kwargs[k]")

    def courbe_koch(self:'Koch'):
        """Dessine une courbe de Koch itérée n fois et de longueur 
        initiale d. Si l'angle est spécifié, dessine une fractale de 
        Cesaro correspondante."""
        direction = (1-2*self.inverse)*self.angle
        self.d = 200 if self.d is None else self.d
        def c_k(n=self.n, length=self.d):
            if n == 0: fd(length)
            else:
                l = length/(2*(math.cos(self.angle)+1))
                c_k(n-1, l), lt(direction), c_k(n-1, l), rt(direction)
                rt(direction), c_k(n-1, l), lt(direction), c_k(n-1, l)
        if π/3 <= self.angle < π/2: return c_k()
            
    def flocon_koch(self:'Koch', carre=False):
        """Dessine un flocon de Koch itéré n fois et de côté initial 
        d."""
        self.d = 300 if self.d is None else self.d
        up(), home(), lt((10-carre)*π/12)
        fd(self.d*sq(2)/2 if carre else self.d*sq(3)/3), down()
        rt((10-carre)*π/12)
        for i in range(3 + carre):
            self.courbe_koch()
            rt((4-carre) * π/6)

    def courbe_90_koch(self:'Koch', typ=1):
        """Dessine une quadratique de Koch itérée n fois, de longueur 
        initiale d, et de type 1 ou 2 (par défaut 1)."""
        self.d = 800 if self.d is None else self.d
        π_2: float = π/2
        def _90_1_(n:int=self.n, length:real=self.d):
            if n==0: fd(length)
            else:
                l: real = length/3
                _90_(n-1, l), lt(π_2), _90_(n-1, l), rt(π_2)
                _90_(n-1, l), rt(π_2), _90_(n-1, l), lt(π_2)
                _90_(n-1, l)

        def _90_2_(n:int=self.n, length:real=self.d):
            if n==0: fd(length)
            else:
                l: real = length/4
                _90_(n-1, l), lt(π_2), _90_(n-1, l), rt(π_2)
                _90_(n-1, l), rt(π_2), _90_(n-1, l)
                _90_(n-1, l), lt(π_2), _90_(n-1, l), lt(π_2)
                _90_(n-1, l), rt(π_2), _90_(n-1, l)
        up()
        if typ==1: 
            goto(-self.width//2, 20-self.height//2)
            _90_: Callable[[int, real], None] = _90_1_
        else:
            goto(-self.d//2, 0)
            _90_: Callable[[int, real], None] = _90_2_
        down()
        _90_()
        
    def terdragon(self:'Koch'):
        self.d = 250 if self.d is None else self.d
        """Dessine une courbe "terdragon" itérée n fois et de longueur
        initiale d"""
        def ter_dragon(n:int=self.n, length:real=self.d):
            _2π_3_ : real = 2*π/3
            if n==0: fd(length)
            else:
                print(n)
                l: real = length/sq(3)
                ter_dragon(n-1, l), rt(_2π_3_), ter_dragon(n-1, l)
                lt(_2π_3_), ter_dragon(n-1, l)
        up(), home(), down(), ter_dragon()

    def gosper(self:'Koch'):
        self.d = 10 if self.d is None else self.d
        π_3 : float = π/3
        π_2 : float = π/2
        _2π_3_ : float = 2*π/3
        def go_spurs_a(n:int=self.n, length:real=self.d*sq(7)/3):
            if n==0: fd(length)
            else:
                l: float = self.d/sq(7) 
                go_spurs_a(n-1, l), rt(π_3), go_spurs_b(n-1, l)
                rt(_2π_3_), go_spurs_b(n-1, l), lt(π_3)
                go_spurs_a(n-1, l), lt(_2π_3_), go_spurs_a(n-1, l)
                go_spurs_a(n-1, l), lt(π_3), go_spurs_b(n-1, l), rt(π_3)

        def go_spurs_b(n:int, length:real):
            if n==0: fd(length)
            else:
                l: float = self.d/sq(7) 
                lt(π_3), go_spurs_a(n-1, l), rt(π_3), go_spurs_b(n-1, l)
                go_spurs_b(n-1, l), rt(_2π_3_), go_spurs_b(n-1, l)
                rt(π_3), go_spurs_a(n-1, l), lt(_2π_3_)
                go_spurs_a(n-1, l), lt(π_3), go_spurs_b(n-1, l)
        up(), goto(-self.d/2, 0), seth(π_2), down()
        return go_spurs_a()

    def arbre_pythagore(self):
        self.d = 100 if self.d is None else self.d
        def pythag(n:int=self.n, d:int=self.d):
            for i in range(4):
                begin_fill(), fd(self.d), rt(π/2), end_fill()
            if n-1>0:
                lt(self.α+π/2), pythag(n-1, self.d*math.cos(self.α))
                rt(π/2), fd(self.d*sin(self.α)), rt(π/2 + self.α)
                fd(self.d), rt(π), fd(self.d)
        up(), home(), goto(-self.d/2,-height/6), down()
        color('black','green'), pythag()
            
    def cercle(rayon,pos=(0,0),lst=[],fill=False):
        up(), goto((pos[0].real,pos[1].real)), seth(0)
        fd(abs(rayon.real)), seth(π/2), down()
        if fill: begin_fill()
        circle(abs(rayon.real))
        if fill: end_fill()
        lst.append((complex(pos[0],pos[1]),abs(rayon.real)))

    def cercle_appollonius(self):
        self.d = 240 if self.d is None else self.d
        C=[(0j, self.d)]
        def verif_liste(lst0, z, r):
            for z0, r0 in lst0:
                if ((abs(z0-z)<(abs(r0)+abs(r))-0.1 and r0!=d) 
                    or (abs(z0-z)<=abs(r0) and r0!=d) 
                    or (abs(z0-z)<=abs(r0) and r0!=d) or z==z0
                    ):
                    return False
            return True
        def descartes(lst,i):
            if i>0:
                length = len(lst)+0
                color('',[
                    'red','magenta','blue','cyan','green','yellow'
                    ][-i%6])
                for z1,r1 in lst[:length]:
                    for z2,r2 in lst[:length]:
                        for z3,r3 in lst[:length]:
                            if z1==z2 or z2==z3 or z3==z1:
                                continue
                            else:
                                k1=(1/r1).real 
                                k2=(1/r2).real
                                k3=(1/r3).real
                                if r1==d:
                                    k1*=-1
                                elif r2==d:
                                    k2*=-1
                                elif r3==d:
                                    k3*=-1
                                k4=k1+k2+k3+2*sqrt(k1*k2+k2*k3+k3*k1)
                                z4_0=(
                                    z1*k1+z2*k2+z3*k3
                                    - 2*sqrt(
                                        z1*z2*k1*k2
                                        + z2*z3*k2*k3
                                        + z3*z1*k3*k1
                                        )
                                    )/k4
                                z4_1=(
                                    z1*k1+z2*k2+z3*k3
                                    + 2*sqrt(
                                        z1*z2*k1*k2 
                                        + z2*z3*k2*k3
                                        + z3*z1*k3*k1
                                        )
                                    )/k4
                                if (
                                    abs((1/k4).real)<abs(d) 
                                    and (z4_0.real, z4_0.imag)!=(0,0) 
                                    and (z4_1.real, z4_1.imag)!=(0,0)
                                    ):
                                    begin_fill()
                                    if (
                                        abs(z4_0)<d 
                                        and verif_liste(
                                            lst[:length],z4_0,1/k4
                                            ) 
                                        and (z4_0,abs(1/k4)) not in lst
                                        ):
                                        cercle(
                                            1/k4, (z4_0.real,z4_0.imag),
                                            lst, True
                                            )
                                    if (
                                        abs(z4_1)<d 
                                        and verif_liste(
                                            lst[:length],z4_1,1/k4
                                            ) 
                                        and (z4_1,abs(1/k4)) not in lst
                                        ):
                                        cercle(
                                            1/k4, (z4_1.real,z4_1.imag),
                                            lst, True
                                            )
                                    end_fill()
                lst=list(set(lst))
                descartes(lst,i-1)
        cercle(d)
        r=d*2*sqrt(3)/(3+2*sqrt(3))
        color('','red')
        for i in range(3):
            cercle(
                r*sqrt(3)/2, (r*(exp(2j*i/3*π).real), 
                r*(exp(2j*i/3*π).imag)), C, True
                )
        cercle(d-r*sqrt(3),(0,0),C,True)
        descartes(C,n)    

    def triangle_sierpinski(self):
        """Dessine un triangle de Sierpinski itéré n fois de côté d """
        self.d = 400 if self.d is None else self.d
        def triangle_blanc(n:int=self.n, d:int=self.d):
            if n>0:
                seth(2*π/3)
                x,y=xcor(),ycor()
                up(), goto(x,y-d*sq(3)/4), down(), begin_fill()
                for i in range(3): fd(d/2), rt(2*π/3)
                end_fill(), up(), goto(x,y+d*sq(3)/8), down()
                triangle_blanc(n-1, d/2), up(), goto(x-d/4,y-d*sq(3)/8)
                down(), triangle_blanc(n-1, d/2), up()
                goto(x+d/4,y-d*sq(3)/8), down(), triangle_blanc(n-1, d/2)
        up(), goto(0,self.d*sq(3)/4), seth(π/3), down(), color("black")
        begin_fill()
        for i in range(3): rt(2*π/3), fd(self.d)
        end_fill(), seth(0), home(), color("white")
        return triangle_blanc()

    def tapis_sierpinski(self):
        """Dessine un tapis de Sierpinski itéré n fois de côté d """
        self.d = 400 if self.d is None else self.d
        def carre_blanc(n:int=self.n, d:real=self.d):
            if n>0:
                seth(0)
                x,y=xcor(),ycor()
                goto(x-d/6,y+d/6), down(), begin_fill()
                for i in range(4): fd(d/3), rt(π/2)
                end_fill(), up()
                for dx in range(-1,2):
                    for dy in range(-1,2):
                        if not (dx or dy): continue
                        else:
                            goto(x+d*dx/3,y+d*dy/3), down()
                            carre_blanc(n-1, d/3), up()
        up(), goto(-self.d/2,self.d/2), seth(0), down()
        color("black","white"), begin_fill()
        for i in range(4): fd(self.d), rt(π/2)
        end_fill(), up(), home(), color("black")
        return carre_blanc()
