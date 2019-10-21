# Poisson solvers

## The elliptic problem for the pressure

The pressure field is obtained by taking the divergence of the horizontal component of the momentum equation
\eqref{eq:momentumStar} and invoking the vertical component to yield an elliptic Poisson equation for the
non-hydrostatic kinematic pressure
```math
\nabla^2\phi_{NH} = \frac{\nabla \cdot \bm{u}^n}{\Delta t} + \nabla \cdot \bm{G}_{\bm{u}} \equiv \mathscr{F}
```
along with homogenous Neumann boundary conditions $\bm{u} \cdot \bm{\hat{n}} = 0$ (Neumann on $\phi$ for wall-bounded
directions and periodic otherwise) and where $\mathscr{F}$ denotes the source term for the Poisson equation.

## Direct method

Discretizing elliptic problems that can be solved via a classical separation-of-variables approach, such as Poisson's
equation, results in a linear system of equations $M\bm{x} = \bm{y}$ where $M$ is a real symmetric matrix of block
tridiagonal form. This allows for the matrix to be decomposed and solved efficiently, provided that the eigenvalues and
eigenvectors of the blocks are known \citep[\S2]{Buzbee70}. In the case of Poisson's equation on a rectangle,
\citet{Hockney65} has taken advantage of the fact that the fast Fourier transform can be used to perform the matrix
multiplication steps resulting in an even more efficient method. \citet{Schumann88} describe the implementation of such
an algorithm for Poisson's equation on a staggered grid with Dirchlet, Neumann, and periodic boundary conditions.

The method can be explained easily by taking the Fourier transform of both sides of \eqref{eq:poisson-pressure} to yield
```math
-(k_x^2 + k_y^2 + k_z^2) \widehat{\phi}_{NH} = \widehat{\mathscr{F}}
\quad \implies \quad
\widehat{\phi}_{NH} = - \frac{\widehat{\mathscr{F}}}{k_x^2 + k_y^2 + k_z^2}
```
where $\widehat{\cdot}$ denotes the Fourier component. Here $k_x$, $k_y$, and $k_z$ are the wavenumbers. However, when
solving the equation on a staggered grid we require a solution for $\phi_{NH}$ that is second-order accurate such that
when when its Laplacian is computed, $\nabla^2\phi_{NH}$ matches $\mathscr{F}$ to machine precision. This is crucial to
ensure that the projection step in \S\ref{sec:fractional-step} works. To do this, the wavenumbers are replaced by
eigenvalues $\lambda_x$, $\lambda_y$, and $\lambda_z$ satisfying the discrete form of Poisson's equation with
appropriate boundary conditions. Thus, Poisson's equation's is diagonalized in Fourier space and the Fourier
coefficients of the solution are easily solved for
```math
\widehat{\phi}_{NH}(i, j, k) = - \frac{\widehat{\mathscr{F}}(i, j, k)}{\lambda^x_i + \lambda^y_j + \lambda^z_k}
```

The eigenvalues are given by \citet{Schumann88} and can also be tediously derived by plugging in the definition of the
discrete Fourier transform into \eqref{eq:poisson-spectral}
```math
\begin{aligned}
    \lambda^x_i &= 4\frac{N_x^2}{L_x^2} \sin^2 \left [ \frac{(i-1)\pi}{N_x}  \right ], \quad i=0,1, \dots,N_x-1 \\
    \lambda^x_j &= 4\frac{N_y^2}{L_y^2} \sin^2 \left [ \frac{(j-1)\pi}{N_y}  \right ], \quad j=0,1, \dots,N_y-1 \\
    \lambda^x_k &= 4\frac{N_z^2}{L_z^2} \sin^2 \left [ \frac{(k-1)\pi}{2N_z} \right ], \quad k=0,1, \dots,N_z-1
\end{aligned}
```
where $\lambda_x$ and $\lambda_y$ correspond to periodic boundary conditions in the horizontal and $\lambda_z$ to
Neumann boundary conditions in the vertical.

There is also an ambiguity in the solution to Poisson's equation as it's only defined up to a constant. To resolve this
we choose the solution with zero mean by setting the zeroth Fourier coefficient $\phi_{000}$ (corresponding to
$k_x = k_y = k_z = 0$) to zero. This also has the added benefit of discarding the zero eigenvalue so we don't divide by
it.

The Fast Fourier transforms are computed using FFTW.jl \citep{Frigo98,Frigo05} on the CPU and using the cuFFT
library on the GPU. Along wall-bouded dimensions, the cosine transform is used. In particular, as the transforms are
performed on a staggered grid, DCT-II (`REDFT10`) is used to perform the forward cosine transform and DCT-III
(`REDFT01`) is used to perform the inverse cosine transform.

## Direct method with a vertically stretched grid

Using Fourier transforms for all three dimensions results in a method requiring $\mathcal{O}(N \log_2 N)$ operations
where $N$ is the total number of grid points. This algorithm can be made even more efficient by solving a tridiagonal
system along one of the dimensions and utilizing cyclic reduction. This results in the *Fourier analysis cyclic
reduction* or $\text{FACR}(\ell)$ algorithm (with $\ell$ cyclic reduction steps) which requires only
$\mathcal{O}(N \log_2\log_2 N)$ operations provided the optimal number of cyclic reduction steps is taken, which is
$\ell = \log_2 \log_2 n$ where $n$ is the number of grid points in the cyclic reduction dimension. The FACR algorithm
was first developed by \citet{Hockney69} and is well reviewed by \citet{Swarztrauber77} then further benchmarked and
extended by \citet{Temperton79} and \citet{Temperton80}.

Furthermore, the FACR algorithm removes the restriction that the grid is uniform in one of the dimensions so it can
be utilized to implement a fast Poisson solver for vertically stretched grids if the cyclic reduction is applied in the
along the vertical dimension.

Expanding $\phi_{NH}$ and $\mathscr{F}$ into Fourier modes along the $x$ and $y$ directions
```math
\phi_{ijk} = \sum_{m=1}^{N_x} \sum_{n=1}^{N_y} \tilde{\phi}_{mnk} \; e^{-i2\pi im / N_x} \;  e^{-i2\pi jn / N_y}
```
and recalling that Fourier transforms do $\partial_x \rightarrow ik_x$ and $\partial_y \rightarrow ik_y$ we can write
\eqref{eq:poisson-pressure} as
```math
\sum_{m=1}^{N_x} \sum_{n=1}^{N_y}
\left\lbrace
    \partial_z^2 \tilde{\phi}_{mnk} - (k_x^2 + k_y^2) \tilde{\phi}_{mnk} - \tilde{\mathscr{F}}_{mnk}
\right\rbrace e^{-i2\pi im / N_x}  e^{-i2\pi jn / N_y} = 0
```
Discretizing the $\partial_z^2$ derivative and equating the term inside the brackets to zero we arrive at
$N_x\times N_y$ symmetric tridiagonal systems of $N_z$ linear equations for the Fourier modes:
```math
\frac{\tilde{\phi}_{mn,k-1}}{\Delta z^F_{k-1}}
- \left\lbrace \frac{1}{1/\Delta z^F_{k-1} + 1/\Delta z^F_k} + \Delta z^C_k (k_x^2 + k_y^2) \right\rbrace
  \tilde{\phi}_{mnk}
+ \frac{\tilde{\phi}_{mn,k+1}}{\Delta z^F_k}
= \Delta z^C_k \tilde{\mathscr{F}}_{mnk}
```

## Cosine transforms on the GPU

Unfortunately cuFFT does not provide cosine transforms and so we must write our own fast cosine transforms for the GPU.
We implemented the fast 1D and 2D cosine transforms described by \citet{Makhoul80} which compute it by applying the
regular Fourier transform to a permuted version of the array.

In this section we will be using the DCT-II as the definition of the forward cosine transform for a real signal of
length $N$
```math
  \text{DCT}(X): \quad Y_k = 2 \sum_{j=0}^{N-1} \cos \left[ \frac{\pi(j + \frac{1}{2})k}{N} \right] X_j
```
and the DCT-III as the definition of the inverse cosine transform
```math
  \text{IDCT}(X): \quad Y_k = X_0 + 2 \sum_{j=1}^{N-1} \cos \left[ \frac{\pi j (k + \frac{1}{2})}{N} \right] X_j
```
and will use $\omega_M = e^{-2\pi i/M}$ to denote the $M^\text{th}$ root of unity, sometimes called the twiddle factors
in the context of FFT algorithms.

### 1D fast cosine transform
To calculate \eqref{eq:FCT} using the fast Fourier transform, we first permute the input signal along the appropriate
dimension by ordering the odd elements first followed by the even elements to produce a permuted signal
```math
    X^\prime_n =
    \begin{cases}
        \displaystyle X_{2N}, \quad 0 \le n \le \left[ \frac{N-1}{2} \right] \\
        \displaystyle X_{2N - 2n - 1}, \quad \left[ \frac{N+1}{2} \right] \le n \le N-1
    \end{cases}
```
where $[a]$ indicates the integer part of $a$. This should produce, for example,
```math
    (a, b, c, d, e, f, g, h) \quad \rightarrow \quad (a, c, e, g, h, f, d, b)
```
after which \eqref{eq:FCT} is computed using
```math
  Y = \text{DCT}(X) = 2 \text{Re} \left\lbrace \omega_{4N}^k \text{FFT} \lbrace X^\prime \rbrace \right\rbrace
```

### 1D fast inverse cosine transform
The inverse \eqref{eq:IFCT} can be computed using
```math
  Y = \text{IDCT}(X) = \text{Re} \left\lbrace \omega_{4N}^{-k} \text{IFFT} \lbrace X \rbrace \right\rbrace
```
after which the inverse permutation of \eqref{eq:permutation} must be applied.

### 2D fast cosine transform
Unfortunately, the 1D algorithm cannot be applied dimension-wise so the 2D algorithm is  more complicated. Thankfully
though, the permutation \eqref{eq:permutation} can be applied dimension-wise. The forward cosine transform for a real
signal of length $N_1 \times N_2$ is then given by
```math
Y_{k_1, k_2} = \text{DCT}(X_{n_1, n_2}) =
2 \text{Re} \left\lbrace
    \omega_{4N_1}^k \left( \omega_{4N_2}^k \tilde{X} + \omega_{4N_2}^{-k} \tilde{X}^- \right)
\right\rbrace
```
where $\tilde{X} = \text{FFT}(X^\prime)$ and $\tilde{X}^-$ indicates that $\tilde{X}$ is indexed in reverse.

### 2D fast inverse cosine transform
The inverse can be computed using
```math
Y_{k_1, k_2} = \text{IDCT}(X_{n_1, n_2}) =
\frac{1}{4} \text{Re} \left\lbrace
    \omega_{4N_1}^{-k} \omega_{4N_2}^{-k}
    \left( \tilde{X} - M_1 M_2 \tilde{X}^{--} \right)
    - i \left( M_1 \tilde{X}^{-+} + M_2 \tilde{X}^{+-} \right)
\right\rbrace
```
where $\tilde{X} = \text{IFFT}(X)$ here, $\tilde{X}^{-+}$ is indexed in reverse along the first dimension,
$\tilde{X}^{-+}$ along the second dimension, and ``\tilde{X}^{--}`` along both. $M_1$ and $M_2$ are masks of lengths
$N_1$ and $N_2$ respectively, both containing ones except at the first element where $M_0 = 0$. Afterwards, the inverse
permutation of \eqref{eq:permutation} must be applied.

Due to the extra steps involved in calculating the cosine transform in 2D, running with two wall-bounded dimensions
typically slows the model down by a factor of 2. Switching to the FACR algorithm may help here as a 2D cosine transform
won't be neccessary anymore.
