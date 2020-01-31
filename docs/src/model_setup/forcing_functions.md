# Forcing functions
Can be used to implement anything you wish, as long as it can be expressed as extra terms in the momentum equation or
tracer evolution equations.

Forcing functions will be called with the signature
```
f(i, j, k, grid, t, U, C, p)
```
where `i, j, k` is the grid index, `grid` is `model.grid`, `t` is the `model.clock.time`, `U` is the named tuple
`model.velocities`, `C` is the named tuple `C.tracers`, and `p` is the user-defined `model.parameters`.

Once you have defined all the forcing functions needed by the model, `ModelForcing` can be used to create a named tuple
of forcing functions that can be passed to the `Model` constructor.

Some examples:

1. Implementing a sponge layer at the bottom of the domain that damps the velocity (to filter out waves) with an
e-folding length scale of 1% of the domain height.
```@example
using Oceananigans # hide
N, L = 16, 100
grid = RegularCartesianGrid(size=(N, N, N), length=(L, L, L))

const τ⁻¹ = 1 / 60  # Damping/relaxation time scale [s⁻¹].
const Δμ = 0.01L    # Sponge layer width [m] set to 1% of the domain height.
@inline μ(z, Lz) = τ⁻¹ * exp(-(z+Lz) / Δμ)

@inline Fu(i, j, k, grid, t, U, C, p) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(i, j, k, grid, t, U, C, p) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(i, j, k, grid, t, U, C, p) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]

forcing = ModelForcing(u=Fu, v=Fv, w=Fw)
model = Model(grid=grid, forcing=forcing)
nothing # hide
```

2. Implementing a point source of fresh meltwater from ice shelf melting via a relaxation term
```@example
using Oceananigans # hide
Nx = Ny = Nz = 16
Lx = Ly = Lz = 1000
grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

λ = 1/(1minute)  # Relaxation timescale [s⁻¹].

# Temperature and salinity of the meltwater outflow.
T_source = -1
S_source = 33.95

# Index of the point source at the middle of the southern wall.
source_index = (Int(Nx/2), 1, Int(Nz/2))

# Point source
@inline T_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.T[i, j, k] - p.T_source), 0)

@inline S_point_source(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse((i, j, k) == p.source_index, -p.λ * (C.S[i, j, k] - p.S_source), 0)

params = (source_index=source_index, T_source=T_source, S_source=S_source, λ=λ)

forcing = ModelForcing(T=T_point_source, S=S_point_source)
```

3. You can also define a forcing as a function of `(x, y, z, t)` or `(x, y, z, t, params)` using the `SimpleForcing`
constructor.

```@example
using Oceananigans # hide
const a = 2.1
fun_forcing(x, y, z, t) = a * exp(z) * cos(t)
u_forcing = SimpleForcing(fun_forcing)

parameterized_forcing(x, y, z, t, p) = p.μ * exp(z/p.λ) * cos(p.ω*t)
v_forcing = SimpleForcing(parameterized_forcing, parameters=(μ=42, λ=0.1, ω=π))

forcing = ModelForcing(u=u_forcing, v=v_forcing)

grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
model = Model(grid=grid, forcing=forcing)
nothing # hide
```
