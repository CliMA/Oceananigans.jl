# Coordinate system and notation

Oceananigans.jl is formulated in a Cartesian coordinate system
$\bm{x} = (x, y, z)$ with unit vectors $\bm{\hat x}$, $\bm{\hat y}$, and $\bm{\hat z}$,
where $\bm{\hat x}$ points east, $\bm{\hat y}$ points north, and $\bm{\hat z}$ points 'upward',
opposite the direction of gravitational acceleration.
We denote time with $t$, partial derivatives with respect to time $t$ or a coordinate $x$
with $\partial_t$ or $\partial_x$, and denote the gradient operator
$\bm{\nabla} \equiv \partial_x \bm{\hat x} + \partial_y \bm{\hat y} + \partial_z \bm{\hat z}$.
We use $u$, $v$, and $w$ to denote the east, north, and vertical velocity components,
such that $\bm{u} = u \bm{\hat x} + v \bm{\hat y} + w \bm{\hat z}$.
