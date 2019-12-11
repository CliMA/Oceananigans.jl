# Model setup

This section describes all the options and features that can be used to set up a model. For more detailed information
consult the API documentation.

Each structure covered in this section can be constructed and passed to the `Model` constructor. For examples of model
construction, see the examples. The verification experiments provide more advanced examples.

## Architecture
Passing `architecture = CPU()` or `architecture = GPU()` to the `Model` constructor will determine whether the model
is time stepped on a CPU or GPU.

Ideally a set up or simulation script does not need to be modified to run on a GPU but we are still smoothing out
rough edges. Generally the CPU wants `Array` objects while the GPU wants `CuArray` objects.

!!! tip "Running on GPUs"
    If you are having issues with running Oceananigans on a GPU, please
    [open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new) and we'll do our best to help out.

## Number type
Passing `float_type=Float64` or `float_type=Float32` to the `Model` constructor causes the model to store all numbers
with 64-bit or 32-bit floating point precision.

!!! note "Avoiding mixed-precision operations"
    When not using `Float64` be careful to not mix different precisions as it could introduce implicit type conversions
    which can negatively effect performance. You can pass the number type desires to many constructors to enforce
    the type you want: e.g. `RegularCartesianGrid(Float32; size=(16, 16, 16), length=(1, 1, 1))` and
    `ConstantIsotropicDiffusivity(Float16; κ=1//7, ν=2//7)`.

!!! warning "Effect of floating point precision on simulation accuracy"
    While we run many tests with both `Float32` and `Float64` it is not clear whether `Float32` is precise enough to
    provide similar accuracy in all use cases. If accuracy is a concern, stick to `Float64`.

    We will be actively investigating the possibility of using lower precision floating point numbers such as `Float32`
    and `Float16` for fluid dynamics as well as the use of alternative number types such as Posits and Sonums.

## Grids
Currently only a regular Cartesian grid with constant grid spacings is supported. The spacing can be different for each
dimension.

When constructing a `RegularCartesianGrid` the number of grid points (or size of the grid) must be passed a tuple along
with the physical length of each dimension.

A regular Cartesian grid with $N_x \times N_y \times N_z = 64 \times 32 \times 16$ grid points and a length of
$L_x = 200$ meters, $L_y = 100$ meters, and $L_z = 100$ meters is constructed using
```@example
grid = RegularCartesianGrid(size=(64, 32, 16), length=(200, 100, 100))
```

!!! info "Default domain"
    By default $x \in [0, L_x]$, $y \in [0, L_y]$, and $z \in [-L_z, 0]$ which is common for oceanographic applications.

### Specifying the domain
To specify a different domain, the `x`, `y`, and `z` keyword arguments can be used instead of `length`. For example,
to use the domain $x \in [-100, 100]$ meters, $y \in [-50, 50]$ meters, and $z \in [0, 100]$ meters
```@example
grid = RegularCartesianGrid(size=(64, 32, 16), x=(-100, 100), y=(-50, 50), z=(0, 100))
```

### Two-dimensional grids
Two-dimensional grids can be constructed by setting the number of grid points along the flat dimension to be 1. A
two-dimensional grid in the $xz$-plane can be constructed using
```@example
grid = RegularCartesianGrid(size=(64, 1, 16), length=(200, 1, 100))
```

In this case the length of the $y$ dimension must be specified but does not matter so we just set it to 1.

2D grids can be used to simulate $xy$, $xz$, and $yz$ planes.

### One-dimensional grids
One-dimensional grids can be constructed in a similar manner, most commonly used to set up vertical column models. For
example, to set up a 1D model with $N_z$ grid points
```@example
grid = RegularCartesianGrid(size=(1, 1, 90), length=(1, 1, 1000))
```

!!! warning "One-dimensional horizontal models"
    We only test one-dimensional vertical models and cannot guarantee that one-dimensional horizontal models will work
    as expected.

## Clock
The clock holds the current iteration number and time. By default the model starts at iteration number 0 and time 0
```@example
clock = Clock(0, 0)
```
but can be modified if you wish to start the model clock at some other time. If you want iteration 0 to correspond to
$t = 3600$ seconds, then you can construct
```#@example
clock = Clock(0, 3600)
```
and pass it to the model.

## Coriolis
The Coriolis option determines whether the fluid experiences the effect of the Coriolis force, or rotation. Currently
three options are available: no rotation, $f$-plane, and $\beta$-plane.

!!! info "Coriolis vs. rotation"
    If you are wondering why this option is called "Coriolis" it is because rotational effects could include the
    Coriolis and centripetal forces, both of which arise in non-inertial reference frames. But here the model only
    considers the Coriolis force.

To use no rotation, pass
```@example
coriolis = nothing
```

### $f$-plane

To set up an $f$-plane with, for example, rotation rate $f = 10^{-4} \text{s}^{-1}$
```@example
coriolis = FPlane(f=1e-4)
```

An $f$-plane can also be specified at some latitude on a spherical planet with a planetary rotation rate. For example,
to specify an $f$-plane at a latitude of $\varphi = 45°\text{N}$ on Earth which has a rotation rate of
$\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$
```@example
coriolis = FPlane(rotation_rate=7.292115e-5, latitude=45)
```
in which case the value of $f$ is given by $2\Omega\sin\varphi$.

### $\beta$-plane
To set up a $\beta$-plane the background rotation rate $f_0$ and the $\beta$ parameter must be specified. For example,
a $\beta$-plane with $f_0 = 10^{-4} \text{s}^{-1}$ and $\beta = 1.5 \times 10^{-11} \text{s}^{-1}\text{m}^{-1}$ can be
set up with
```@example
coriolis = BetaPlane(f₀=1e-4, β=1.5e-11)
```

Alternatively, a $\beta$-plane can also be set up at some latitude on a spherical planet with a planetary rotation rate
and planetary radius. For example, to specify a $\beta$-plane at a latitude of $\varphi = 10\degree{S}$ on Earth
which has a rotation rate of $\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$ and a radius of $R = 6,371 \text{km}$
```@example
coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=-10, radius=6371e3)
```
in which case $f_0 = 2\Omega\sin\varphi$ and $\beta = 2\Omega\cos\varphi / R$.

## Tracers

The tracers to be advected around can be specified via a list of symbols. By default the model evolves temperature and
salinity
```@example
tracers = (:T, :S)
```
but any number of arbitrary tracers can be appended to this list. For example, to evolve quantities $C_1$, CO₂, and
nitrogen as passive tracers you could set them up as
```@example
tracers = (:T, :S, :C₁, :CO₂, :nitrogen)
```

!!! info "Active vs. passive tracers"
    An active tracer typically denotes a tracer quantity that affects the fluid dynamics through buoyancy. In the ocean
    temperature and salinity are active tracers. Passive tracers, on the other hand, typically do not affect the fluid
    dynamics are are _passively_ advected around by the flow field.

## Buoyancy and equation of state
The buoyancy option selects how buoyancy is treated. There are currently three options:
1. No buoyancy (and no gravity).
2. Evolve buoyancy as a tracer.
3. _Seawater buoyancy_: evolve temperature $T$ and salinity $S$ as tracers with a value for the gravitational
   acceleration $g$ and an appropriate equation of state.

### No buoyancy
To turn off buoyancy (and gravity) simply pass
```@example
buoyancy = nothing
```
to the `Model` constructor. In this case, you will probably also want to explicitly specify which tracers to evolve.
In particular, you probably will not want to evolve temperature and salinity, which are included by default. To specify
no tracers, also pass
```@example
tracers = ()
```
to the `Model` constructor.

### Buoyancy as a tracer
To directly evolve buoyancy as a tracer simply pass
```@example
buoyancy = BuoyancyTracer()
```
to the `Model` constructor. Buoyancy `:b` must be included as a tracer, for example, by also passing
```@example
tracers = (:b)
```

### Seawater buoyancy
To evolve temperature $T$ and salinity $S$ and diagnose the buoyancy, you can pass
```@example
buoyancy = SeawaterBuoyancy()
```
which is also the default. Without any options specified, a value of $g = 9.80665 \; \text{m/s}^2$ is used for the
gravitational acceleration (corresponding to [standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)) along
with a linear equation of state with thermal expansion and haline contraction coefficients suitable for water.

If, for example, you wanted to simulate fluids on another planet such as Europa where $g = 1.3 \; \text{m/s}^2$, then
use
```@example
buoyancy = SeawaterBuoyancy(gravitational_acceleration=1.3)
```

When using `SeawaterBuoyancy` temperature `:T` and salinity `:S` tracers must be specified
```@example
tracers = (:T, :S)
```

#### Linear equation of state
To use non-default thermal expansion and haline contraction coefficients, say
$\alpha = 2 \times 10^{-3} \; \text{K}^{-1}$ and $\beta = 5 \times 10{-4} \text{ppt}^{-1}$ corresponding to some other
fluid, then use

```@example
buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(α=1.67e-4, β=7.80e-4))
```

#### Idealized nonlinear equation of state
Instead of a linear equation of state, five idealized nonlinear equation of state as described by Roquet et al. (2015)
may be specified. See [`RoquetIdealizedNonlinearEquationOfState`](@ref RoquetIdealizedNonlinearEquationOfState).

## Boundary conditions
A boundary condition is applied to each field, dimension, and endpoint. There are left and right (or bottom and top)
boundary conditions for each of the x, y, and z dimensions so each field is associated with 6 boundary conditions. Each
of these boundary conditions may be specified individually. Each boundary condition can be specified via a constant
value, an array, or a function.

See [Numerical implementation of boundary conditions](@ref numerical_bcs) for more details.

!!! warning "Consistent boundary conditions"
    Be careful to ensure that you don't set up a model with inconsistent boundary conditions. For example, periodic
    boundary conditions should remain imposed on all fields and endpoints for periodic dimensions, and velocities
    normal to a wall (e.g. vertical velocity w with walls at the top and bottom) must have no-penetration boundary
    conditions.

### Types of boundary conditions
1. [`Periodic`](@ref Periodic)
2. [`Flux`](@ref Flux)
3. [`Value`](@ref Value) ([`Dirchlet`](@ref))
4. [`Gradient`](@ref Gradient) ([`Neumann`](@ref))
5. [`No-penetration`](@ref NoPenetration)

Notice that open boundary conditions and radiation boundary conditions can be imposed via flux or value boundary
conditions defined by a function or array.

### Default boundary conditions
By default, periodic boundary conditions are applied on all fields along periodic dimensions. All other boundary
conditions are no-flux, except for velocities normal to a wall which get no-penetration boundary conditions.

### Boundary condition structures
Oceananigans uses a hierarchical structure to expressing boundary conditions.
1. A [`BoundaryCondition`](@ref) is associated with every field, dimension, and endpoint.
2. Boundary conditions specifying the condition at the left and right endpoints (or top and bottom endpoints) are
   grouped into [`CoordinateBoundaryConditions`](@ref).
3. A set of three `CoordinateBoundaryConditions` specifying the boundary conditions along the x, y, and z dimensions
   for a single field are grouped into a [`FieldBoundaryConditions`](@ref) named tuple.
4. A set of `FieldBoundaryConditions`, one for each field, are grouped together into a named tuple and passed to the
   `Model` constructor.

Boundary conditions are defined at model construction time by passing a named tuple of `FieldBoundaryConditions`
specifying boundary conditions on every field: velocities ($u$, $v$, $w$) and all tracers.

Typically you only want to impose a few boundary conditions, in which case it's useful to use convenience constructors
such as [`HorizontallyPeriodicBCs`](@ref) when constructing horizontally periodic boundary conditions for a field and
[`HorizontallyPeriodicSolutionBCs`](@ref) when constructing horizontally periodic boundary conditions for a model.
Also see [`ChannelBCs`](@ref) and [`ChannelSolutionBCs`](@ref).

See the sections below for more details. The examples and verification experiments also provide examples for setting up
many difference kinds of boundary conditions.

### Creating individual boundary conditions
Some examples of creating individual boundary conditions:

1. A constant Value (Dirchlet) boundary condition, perhaps representing a constant temperature at some boundary.
```@example
constant_T_bc = BoundaryCondition(Value, 20)
```

2. A constant flux boundary condition, perhaps representing a constant wind stress at some boundary such as the ocean
   surface.
```@example
ρ₀ = 1027  # Reference density [kg/m³]
τₓ = 0.08  # Wind stress [N/m²]
wind_stress_bc = BoundaryCondition(Flux, τₓ/ρ₀)
```

3. A spatially varying (white noise) cooling flux to be imposed at some boundary. Note that the boundary condition
   is given by the array `Q` here. When running on the GPU, `Q` must be converted to a `CuArray`.
```@example
Nx, Ny = 16, 16  # Number of grid points.

ρ₀ = 1027  # Reference density [kg/m³]
cₚ = 4000  # Heat capacity of water at constant pressure [J/kg/K]

Q  = randn(Nx, Ny) ./ (ρ₀ * cₚ)

white_noise_T_bc = BoundaryCondition(Flux, Q)
```

### Specifying boundary conditions with functions
You can also specify the boundary condition via a function. For z boundary conditions the function will be called with
the signature
```
f(i, j, grid, t, U, C, params)
```
where `i, j` is the grid index, `grid` is `model.grid`, `t` is the `model.clock.time`, `U` is the named tuple
`model.velocities`, `C` is the named tuple `C.tracers`, and `params` is the user-defined `model.parameters`. The
signature is similar for x and y boundary conditions expect that `i, j` is replaced with `j, k` and `i, k` respectively.

We can add a fourth example now:
4. A spatially varying and time-dependent heating representing perhaps a localized source of heating modulated by a
   diurnal cycle.
```@example
@inline Q(i, j, grid, t, U, C, params) = @inbounds exp(-(grid.xC[i]^2 + grid.yC[j]^2)) * sin(2π*t)
localized_heating_bc = BoundaryCondition(Flux, Q)
```

!!! info "Performance of functions in boundary conditions"
    For performance reasons, you should define all functions used in boundary conditions as inline functions via the
    `@inline` macro. If any arrays are accessed within the function, disabling bounds-checking with `@inbounds` can
    also speed things up.

### Specifying boundary conditions on a field
To, for example, create a set of horizontally periodic field boundary conditions
```@example
T_bcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 20),
                                bottom = BoundaryCondition(Gradient, 0.01))
```
which will create a [`FieldBoundaryConditions`](@ref) object for temperature T appropriate for horizontally periodic
model configurations where the x and y boundary conditions are all periodic.

### Specifying model boundary conditions
A named tuple of [`FieldBoundaryConditions`](@ref) objects must be passed to the Model constructor specifying boundary
conditions on all fields. To, for example, impose non-default boundary conditions on the u-velocity and temperature
```@example
u_bcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0.1),
                                bottom = BoundaryCondition(Value, -0.1))
T_bcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 20),
                                bottom = BoundaryCondition(Gradient, 0.01))

model_bcs = HorizontallyPeriodicSolutionBCs(u=u_bc, T=T_bcs)

model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)),
              boundary_conditions=model_bcs)
```

## Forcing functions
Can be used to implement anything you wish, as long as it can be expressed as extra terms in the momentum equation or
tracer evolution equations.

Forcing functions will be called with the signature
```
f(i, j, k, grid, t, U, C, params)
```
where `i, j, k` is the grid index, `grid` is `model.grid`, `t` is the `model.clock.time`, `U` is the named tuple
`model.velocities`, `C` is the named tuple `C.tracers`, and `params` is the user-defined `model.parameters`.

Once you have defined all the forcing functions needed by the model, `ModelForcing` can be used to create a named tuple
of forcing functions that can be passed to the `Model` constructor.

Some examples:

1. Implementing a sponge layer at the bottom of the domain that damps the velocity (to filter out waves) with an
e-folding length scale of 1% of the domain height.
```@example
N, L = 16, 100
grid = RegularCartesianGrid(size=(N, N, N), length=(L, L, L))

const τ⁻¹ = 1 / 60  # Damping/relaxation time scale [s⁻¹].
const Δμ = 0.01L    # Sponge layer width [m] set to 1% of the domain height.
@inline μ(z, Lz) = τ⁻¹ * exp(-(z+Lz) / Δμ)

@inline Fu(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]

forcing = ModelForcing(Fu=Fu, Fv=Fv, Fw=Fw)
model = Model(grid=grid, forcing=forcing)
```

2. Implementing a point source of fresh meltwater from ice shelf melting via a relaxation term
```@example
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
const a = 2.1
fun_forcing(x, y, z, t) = a * exp(z) * cos(t)
u_forcing = SimpleForcing(fun_forcing)

parameterized_forcing(x, y, z, t, p) = p.μ * exp(z/p.λ) * cos(p.ω*t)
v_forcing = SimpleForcing(parameterized_forcing, parameters=(μ=42, λ=0.1, ω=π))

forcing = ModelForcing(u=u_forcing, v=v_forcing)

model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)),
              forcing=forcing)
```

## Parameters
A user-defined object (could be anything) can be passed via the `parameters` keyword to be accessed by forcing functions
and boundary condition functions.

## Turbulent diffusivity closures and large eddy simulation models
A turbulent diffusivty closure representing the effects of viscous dissipation and diffusion can be passed via the
`closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for more details
on turbulent diffusivity closures.

### Constant isotropic diffusivity
To use constant isotropic values for the viscosity ν and diffusivity κ you can use `ConstantIsotropicDiffusivity`
```@example
closure = ConstantIsotropicDiffusivity(ν=1e-2, κ=1e-2)
```
### Constant anisotropic diffusivity
To specify constant values for the horizontal and vertical viscosities, $\nu_h$ and $\nu_v$, and horizontal and vertical
diffusivities, $\kappa_h$ and $\kappa_v$, you can use `ConstantAnisotropicDiffusivity`
```@example
closure = ConstantAnisotropicDiffusivity(νh=1e-3, νv=5e-2, κh=2e-3, κv=1e-1)
```

### Smagorinsky-Lilly
To use the Smagorinsky-Lilly LES closure, no parameters are required
```@example
closure = SmagorinskyLilly()
```
although they may be specified. By default, the background viscosity and diffusivity are assumed to be the molecular
values for seawater. For more details see [`SmagorinskyLilly`](@ref).

### Anisotropic minimum dissipation
To use the constant anisotropic minimum dissipation (AMD) LES closure, no parameters are required
```@example
closure = AnisotropicMinimumDissipation()
```
although they may be specified. By default, the background viscosity and diffusivity are assumed to be the molecular
values for seawater. For more details see [`AnisotropicMinimumDissipation`](@ref).

## Diagnostics
Diagnostics are a set of general utilities that can be called on-demand during time-stepping to compute quantities of
interest you may want to save to disk, such as the horizontal average of the temperature, the maximum velocity, or to
produce a time series of salinity. They also include utilities for diagnosing model health, such as the CFL number or
to check for NaNs.

Diagnostics are stored as a list of diagnostics in `model.diagnostics`. Diagnostics can be specified at model creation
time or be specified at any later time and appended (or assigned with a key value pair) to `model.diagnostics`.

Most diagnostics can be run at specified frequencies (e.g. every 25 time steps) or specified intervals (e.g. every
15 minutes of simulation time). If you'd like to run a diagnostic on demand then do not specify a frequency or interval
(and do not add it to `model.diagnostics`).

We describe the `HorizontalAverage` diagnostic in detail below but see the API documentation for other diagnostics such
as [`Timeseries`](@ref), [`FieldMaximum`](@ref), [`CFL`](@ref), and [`NaNChecker`](@ref).

### Horizontal averages
You can create a `HorizontalAverage` diagnostic by passing a field to the constructor, e.g.
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
T_avg = HorizontalAverage(model.tracers.T)
push!(model.diagnostics, T_avg)
```
which can then be called on-demand via `T_avg(model)` to return the horizontally averaged temperature. When running on
the GPU you may want it to return an `Array` instead of a `CuArray` in case you want to save the horizontal average to
disk in which case you'd want to construct it like
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
T_avg = HorizontalAverage(model.tracers.T; return_type=Array)
push!(model.diagnostics, T_avg)
```

You can also use pass an abstract operator to take the horizontal average of any diagnosed quantity. For example, to
compute the horizontal average of the vertical component of vorticity:
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
ζ_avg = HorizontalAverage(ζ)
model.diagnostics[:vorticity_profile] = ζ_avg
```

See [`HorizontalAverage`](@ref) for more details and options.

## Output writers
Saving model data to disk can be done in a flexible manner using output writers. The two main output writers currently
implemented are a NetCDF output writer (relying on [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl))
and a JLD2 output writer (relying on [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)).

Output writers are stored as a list of output writers in `model.output_writers`. Output writers can be specified at
model creation time or be specified at any later time and appended (or assigned with a key value pair) to
`model.output_writers`.

### NetCDF output writer
Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

The following example shows how to construct NetCDF output writers for two different kinds of outputs (3D fields and
slices) along with output attributes
```@example
Nx = Ny = Nz = 16
model = Model(grid=RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1)))

fields = Dict(
    "u" => model.velocities.u,
    "T" => model.tracers.T
)

output_attributes = Dict(
    "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
    "T" => Dict("longname" => "Temperature", "units" => "C")
)

model.output_writers[:field_writer] = NetCDFOutputWriter(model, fields; filename="output_fields.nc",
                                                         interval=6hour, output_attributes=output_attributes)

model.output_writers[:surface_slice_writer] = NetCDFOutputWriter(model, fields; filename="output_surface_xy_slice.nc",
                                                                 interval=5minute, output_attributes=output_attributes,
                                                                 zC=Nz, zF=Nz)
```

See [`NetCDFOutputWriter`](@ref) for more details and options.

### JLD2 output writer
JLD2 is a an HDF5 compatible file format written in pure Julia and is generally pretty fast. JLD2 files can be opened in
Python with the [h5py](https://www.h5py.org/) package.

The JLD2 output writer is generally used by passing it a dictionary or named tuple of (label, function) pairs where the
functions have a single input `model`. Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file. For example, to write out 3D fields for w and T and a horizontal average
of T every 1 hour of simulation time to a file called `some_data.jld2`
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))

function init_save_some_metadata(file, model)
    file["author"] = "Chim Riggles"
    file["parameters/coriolis_parameter"] = 1e-4
    file["parameters/density"] = 1027
end

T_avg =  HorizontalAverage(model.tracers.T)

outputs = Dict(
    :w => model -> model.velocities.u,
    :T => model -> model.tracers.T,
    :T_avg => model -> T_avg(model)
)

jld2_writer = JLD2OutputWriter(model, outputs; init=init_save_some_metadata, interval=1hour, prefix="some_data")

push!(model.output_writers, jld2_writer)
```

See [`JLD2OutputWriter`](@ref) for more details and options.

### Checkpointer
A checkpointer can be used to serialize the entire model state to a file from which the model can be restored at any
time. This is useful if you'd like to periodically checkpoint when running long simulations in case of crashes or
cluster time limits, but also if you'd like to restore from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 1,000,000 seconds of simulation time to files of
the form `model_checkpoint_xxx.jld2` where `xxx` is the iteration number (automatically filled in)
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
model.output_writers[:checkpointer] = Checkpointer(model; interval=1e6, prefix="model_checkpoint")
```

The default options should provide checkpoint files that are easy to restore from in most cases. For more advanced
options and features, see [`Checkpointer`](@ref).

#### Restoring from a checkpoint file
To restore the model from a checkpoint file, for example `model_checkpoint_12345.jld2`, simply call
```
model = restore_from_checkpoint("model_checkpoint_12345.jld2")
```
which will create a new model object that is identical to the one that was serialized to disk. You can continue time
stepping after restoring from a checkpoint.

You can pass additional parameters to the `Model` constructor. See [`restore_from_checkpoint`](@ref) for more
information.

#### Restoring with functions
JLD2 cannot serialize functions to disk. so if you used forcing functions, boundary conditions containing functions, or
the model included references to functions then they will not be serialized to the checkpoint file. When restoring from
a checkpoint file, any model property that contained functions must be manually restored via keyword arguments to
[`restore_from_checkpoint`](@ref).

## Time stepping
Once you're ready to time step the model simply call
```
time_step!(model; Δt=10)
```
to take a single time step with step size 10. To take multiple time steps also pass an `Nt` keyword argument like
```
time_step!(model; Δt=10, Nt=50)
```

By default, `time_step!` uses a first-order forward Euler time step to take the first time step then uses a second-order
Adams-Bashforth method for the remaining time steps (which required knowledge of the previous time step). If you are
resuming time-stepping then you should not use a forward Euler initialization time step. This can be done via
```
time_step!(model; Δt=10)
time_step!(model; Δt=10, Nt=50, init_with_euler=false)
```

### Adaptive time stepping
Adaptive time stepping can be acomplished using the [`TimeStepWizard`](@ref). It can be used to compute time steps based
on capping the CFL number at some value. You must remember to update the time step every so often. For example, to cap
the CFL number at 0.3 and update the time step every 50 time steps:
```
wizard = TimeStepWizard(cfl=0.3, Δt=1.0, max_change=1.2, max_Δt=30.0)

while model.clock.time < end_time
    time_step!(model; Δt=wizard.Δt, Nt=50)
    update_Δt!(wizard, model)
end
```
See [`TimeStepWizard`](@ref) for documentation of other features and options.

!!! warn "Maximum CFL with second-order Adams-Bashforth time stepping"
    For stable time-stepping it is recommended to cap the CFL at 0.3 or smaller, although capping it at 0.5 works well
    for certain simulations. For some simulations, it may be neccessary to cap the CFL number at 0.1 or lower.

!!! warn "Adaptive time stepping with second-order Adams-Bashforth time stepping"
    You should use an initializer forward Euler time step whenever changing the time step (i.e. `init_with_euler=true`
    which is the default value).

## Setting initial conditions
Initial conditions are imposed after model construction. This can be easily done using the the `set!` function, which
allows the setting of initial conditions using constant values, arrays, or functions.

```@example
set!(model.velocities.u, 0.1)
```

```@example
∂T∂z = 0.01
ϵ(σ) = σ * randn()
T₀(x, y, z) = ∂T∂z * z + ϵ(1e-8)
set!(model.tracers.T, T₀)
```
