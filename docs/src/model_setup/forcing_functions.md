# Forcing functions

Forcing functions can be used to implement anything you wish, as long as it can be expressed as extra terms in the
momentum equation or tracer evolution equations.

Raw forcing functions will be called with the signature

```julia
f(i, j, k, grid, clock, state)
```

where `i, j, k` is the grid index, `grid` is `model.grid`, `clock` is the `model.clock`, and `state` is a named tuple
containing `state.velocities`, `state.tracers`, and `state.diffusivities`.

To include some parameters in the function definition, use a [`ParameterizedForcing`](@ref). The function signature then
becomes

```julia
f(i, j, k, grid, clock, state, parameters)
```

To access grid coordinates instead use a [`SimpleForcing`](@ref) and pass a function with one of four signatures

```julia
f(x, y, z, t)
f(x, y, z, t, field)
f(x, y, z, t, parameters)
f(x, y, z, t, field, parameters)
```

with two additional optional arguments `field` and `parameters` (see the [`SimpleForcing`](@ref) for more details).

Once you have defined all the forcing functions needed by the model, `ModelForcing` can be used to create a named tuple
of forcing functions that can be passed to the model constructor.

Some examples:

```@meta
DocTestSetup = quote
    using Oceananigans
    using Oceananigans.Forcing
end
```

1. Implementing a point source of fresh meltwater from ice shelf melting via a relaxation term

```jldoctest
Nx = Ny = Nz = 16
Lx = Ly = Lz = 1000
grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

λ = 1/60  # Relaxation timescale [s⁻¹].

# Temperature and salinity of the meltwater outflow.
T_source = -1
S_source = 33.95

# Index of the point source at the middle of the southern wall.
source_index = (Int(Nx/2), 1, Int(Nz/2))

params = (source_index=source_index, T_source=T_source, S_source=S_source, λ=λ)

# Point source
@inline T_point_source(i, j, k, grid, clock, state, params) =
    @inbounds ifelse((i, j, k) == params.source_index, -params.λ * (state.tracers.T[i, j, k] - params.T_source), 0)

@inline S_point_source(i, j, k, grid, clock, state, params) =
    @inbounds ifelse((i, j, k) == params.source_index, -params.λ * (state.tracers.S[i, j, k] - params.S_source), 0)

T_forcing = ParameterizedForcing(T_point_source, params)
S_forcing = ParameterizedForcing(S_point_source, params)

forcing = ModelForcing(T=T_forcing, S=S_forcing)
model = IncompressibleModel(grid=grid, forcing=forcing)

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0) 
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
├── tracers: (:T, :S)
├── closure: ConstantIsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```

2. Implementing a sponge layer at the bottom of the domain that damps the velocity (to filter out waves) with an
   e-folding length scale of 1% of the domain height.

```jldoctest
N, L = 16, 100
grid = RegularCartesianGrid(size=(N, N, N), extent=(L, L, L))

τ⁻¹ = 1 / 60  # Damping/relaxation time scale [s⁻¹].
Δμ = 0.01L    # Sponge layer width [m] set to 1% of the domain height.

params = (Lz=grid.Lz, τ⁻¹=τ⁻¹, Δμ=Δμ)

@inline μ(z, p) = p.τ⁻¹ * exp(-(z + p.Lz) / p.Δμ)

@inline Fu(x, y, z, t, u, p) = @inbounds -μ(z, p) * u
@inline Fv(x, y, z, t, v, p) = @inbounds -μ(z, p) * v
@inline Fw(x, y, z, t, w, p) = @inbounds -μ(z, p) * w

u_forcing = SimpleForcing(Fu, field_in_signature=true, parameters=params)
v_forcing = SimpleForcing(Fv, field_in_signature=true, parameters=params)
w_forcing = SimpleForcing(Fw, field_in_signature=true, parameters=params)

forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing)
model = IncompressibleModel(grid=grid, forcing=forcing)

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0) 
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
├── tracers: (:T, :S)
├── closure: ConstantIsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```
