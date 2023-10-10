using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: SeawaterBuoyancy, Zᶜᶜᶜ
using Oceananigans.Fields: field
using Oceananigans.Grids: Center
using SeawaterPolynomials: BoussinesqEquationOfState
import SeawaterPolynomials.ρ

export SeawaterDensity

"Extend `SeawaterPolynomials.ρ` to compute density for a `KernelFunctionOperation` -
**note** `eos` must be `BoussinesqEquationOfState` because a reference density is needed for the computation."
@inline ρ(i, j, k, grid, eos, T, S, Z) = @inbounds ρ(T[i, j, k], S[i, j, k], Z[i, j, k], eos)

"Return a `KernelFunctionOperation` to compute the in-situ `seawater_density`."
seawater_density(grid, eos, temperature, salinity, geopotential_height) =
    KernelFunctionOperation{Center, Center, Center}(ρ, grid, eos, temperature, salinity, geopotential_height)

const ModelsWithBuoyancy = Union{NonhydrostaticModel, HydrostaticFreeSurfaceModel}

validate_model_eos(eos) = eos isa BoussinesqEquationOfState ? nothing :
                                                              throw(ArgumentError("The equation of state must be a `BoussinesqEquationOfState` to compute the density."))
# some nice fallbacks
model_temperature(bf, model)     = model.tracers.T
model_salinity(bf, model)        = model.tracers.S
model_geopotential_height(model) = KernelFunctionOperation{Center, Center, Center}(Zᶜᶜᶜ, model.grid)

const ConstantTemperatureSB = SeawaterBuoyancy{FT, EOS, <:Number, <:Nothing} where {FT, EOS}
const ConstantSalinitySB    = SeawaterBuoyancy{FT, EOS, <:Nothing, <:Number} where {FT, EOS}

model_temperature(b::ConstantTemperatureSB, model) = b.constant_temperature
model_salinity(b::ConstantSalinitySB, model)       = b.constant_salinity

"""
    SeawaterDensity(model; temperature, salinity, geopotential_height)

Return a `KernelFunctionOperation` that computes the in-situ density of seawater
with (gridded) `temperature`, `salinity`, and at `geopotential_height`. To compute the
in-situ density, the 55 term polynomial approximation to the equation of state from
[Roquet et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1463500315000566?ref=pdf_download&fr=RR-2&rr=813416acba58557b) is used.
By default the `seawater_density` extracts the geopotential height from the model to compute
the in-situ density. To compute a potential density at some user chosen reference geopotential height,
set `geopotential_height` to a constant for the density computation,

```julia
geopotential_height = 0 # sea-surface height
σ₀ = seawater_density(model; geopotential_height)
```

**Note:** `SeawaterDensity` must be passed a `BoussinesqEquationOfState` to compute the
density. See the [relevant documentation](https://clima.github.io/OceananigansDocumentation/dev/model_setup/buoyancy_and_equation_of_state/#Idealized-nonlinear-equations-of-state)
for how to set `SeawaterBuoyancy` using a `BoussinesqEquationOfState`.

Example
=======

Compute a density `Field` using the `KernelFunctionOperation` returned from `SeawaterDensity`

```jldoctest density1

julia> using Oceananigans, Oceananigans.Models, SeawaterPolynomials.TEOS10

julia> grid = RectilinearGrid(CPU(), size=(32, 32), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
128×128×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── Periodic x ∈ [-7.51279e-18, 6.28319) regularly spaced with Δx=0.0490874
├── Periodic y ∈ [-7.51279e-18, 6.28319) regularly spaced with Δy=0.0490874
└── Flat z

julia> tracers = (:S, :T)
(:S, :T)

julia> eos = TEOS10EquationOfState()
BoussinesqEquationOfState{Float64}:
    ├── seawater_polynomial: TEOS10SeawaterPolynomial{Float64}
    └── reference_density: 1020.0

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
└── equation of state: BoussinesqEquationOfState{Float64}

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 32×32×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: (S, T)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy with g=9.80665 and BoussinesqEquationOfState{Float64} with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> set!(model, S = 34.7, T = 0.5)

julia> density_field = Field(SeawaterDensity(model))
32×32×1 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 32×32×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: ZeroFlux
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 38×38×1 OffsetArray(::Array{Float64, 3}, -2:35, -2:35, 1:1) with eltype Float64 with indices -2:35×-2:35×1:1
    └── max=0.0, min=0.0, mean=0.0

julia> compute!(density_field)
32×32×1 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 32×32×1 RectilinearGrid{Float64, Periodic, Periodic, Flat} on CPU with 3×3×0 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Nothing, top: Nothing, immersed: ZeroFlux
├── operand: KernelFunctionOperation at (Center, Center, Center)
├── status: time=0.0
└── data: 38×38×1 OffsetArray(::Array{Float64, 3}, -2:35, -2:35, 1:1) with eltype Float64 with indices -2:35×-2:35×1:1
    └── max=1027.7, min=1027.7, mean=1027.7

```

Values for `temperature`, `salinity` and `geopotential_height` can be passed to
`SeawaterDensity` to override the defaults that are obtained from the `model`.
"""
function SeawaterDensity(model::ModelsWithBuoyancy;
                         temperature = model_temperature(model.buoyancy.model, model),
                         salinity = model_salinity(model.buoyancy.model, model),
                         geopotential_height = model_geopotential_height(model))

    eos = model.buoyancy.model.equation_of_state
    validate_model_eos(eos)
    # Convert function or constant user input to AbstractField
    grid = model.grid
    loc = (Center, Center, Center)
    temperature = field(loc, temperature, grid)
    salinity = field(loc, salinity, grid)
    # Preferable here is to leave `geopotential_height` as an `AbstractOperation` rather than creating a `Field`
    geopotential_height = geopotential_height isa KernelFunctionOperation ? Field(geopotential_height) :
                                                                            field(loc, geopotential_height, grid)

    return seawater_density(grid, eos, temperature, salinity, geopotential_height)
end
