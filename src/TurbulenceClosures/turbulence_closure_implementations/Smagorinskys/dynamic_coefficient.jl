using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField, Field, compute!, interpolate, xnode, ynode, znode
using Statistics: mean

import Oceananigans: prognostic_state, restore_prognostic_state!

mutable struct DynamicCoefficient{A, FT, S}
    averaging :: A
    minimum_numerator :: FT
    schedule :: S
end

struct DeviceDynamicCoefficient{FT}
    minimum_numerator :: FT
end

const DynamicSmagorinsky = Union{
    Smagorinsky{<:Any, <:DynamicCoefficient},
    Smagorinsky{<:Any, <:DeviceDynamicCoefficient},
}

"""
    DynamicSmagorinsky([time_discretization=ExplicitTimeDiscretization(), FT=Float64;]
                       averaging = LagrangianAveraging(),
                       Pr = 1,
                       schedule = IterationInterval(1),
                       minimum_numerator = 1e-32)

Returns a `Smagorinsky`-type closure with dynamic computation of the Smagorinsky coefficient
according to the scale invariant procedure described by [Bou-Zeid et al. (2005)](@cite BouZeid05).

`DynamicSmagorinsky` requires an `averaging` procedure.
The default is `LagrangianAveraging`, which averages flow characteristics along Lagrangian trajectories).
Another option is a tuple of integers indicating a directional averaging procedure along chosen dimensions
For example, `averaging = (1, 2)` invokes averaging in the ``x, y`` directions.

The coefficient is updated according to `schedule`. Less frequent updates than `IterationInterval(1)`
may be used as a performance optimization for cases where dynamic coefficient computation is relatively expensive.
`minimum_numerator` defines the minimum value that is acceptable in the denominator of the final calculation.

Multi-stage timestepper compatibility
=====================================

For multi-stage timesteppers like `RungeKutta3`, the dynamic coefficient computation is performed
only at the final stage of each iteration (when `clock.stage == 1`).

This is necessary because RK3 calls `update_state!` multiple times per iteration which for
`LagrangianAveraging` would apply the time-average multiple times per iteration with small
fractional Δt values.

Examples
========

```jldoctest smag
julia> using Oceananigans

julia> closure = DynamicSmagorinsky()
DynamicSmagorinsky{Float64}:
├── averaging = Oceananigans.TurbulenceClosures.Smagorinskys.LagrangianAveraging()
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

To compute the dynamic coefficient every 5 time steps, for example, we can use

```jldoctest smag
julia> closure = DynamicSmagorinsky(schedule = IterationInterval(5))
DynamicSmagorinsky{Float64}:
├── averaging = Oceananigans.TurbulenceClosures.Smagorinskys.LagrangianAveraging()
├── schedule = IterationInterval(5, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

For situations that are homogeneous in the ``x``-direction, averaging in ``x`` might provide
cost savings and better averaging properties than a Lagrangian average:

```jldoctest smag
julia> closure = DynamicSmagorinsky(averaging=1)
DynamicSmagorinsky{Float64}:
├── averaging = (1,)
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

or to average in the ``x`` and ``y`` directions:

```jldoctest smag
julia> closure = DynamicSmagorinsky(averaging=(1, 2))
DynamicSmagorinsky{Float64}:
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

References
==========

Bou-Zeid, Elie, Meneveau, Charles, and Parlange, Marc. (2005) A scale-dependent Lagrangian dynamic model for
large eddy simulation of complex turbulent flows, Physics of Fluids, **17**, 025105.
"""
function DynamicSmagorinsky(time_discretization = ExplicitTimeDiscretization(),
                            FT = Oceananigans.defaults.FloatType;
                            averaging = LagrangianAveraging(),
                            Pr = 1,
                            schedule = IterationInterval(1),
                            minimum_numerator = 1e-32)

    coefficient = DynamicCoefficient(FT; averaging, schedule, minimum_numerator)
    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)

    return Smagorinsky{TD}(coefficient, Pr)
end

DynamicSmagorinsky(FT::DataType; kwargs...) = DynamicSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)
Adapt.adapt_structure(to, dc::DynamicCoefficient) = DeviceDynamicCoefficient(dc.minimum_numerator)

const DirectionallyAveragedCoefficient{N} = DynamicCoefficient{<:Union{NTuple{N, Int}, Int, Colon}} where N
const DirectionallyAveragedDynamicSmagorinsky{N} = Smagorinsky{<:Any, <:DirectionallyAveragedCoefficient{N}} where N

struct LagrangianAveraging end
const LagrangianAveragedCoefficient = DynamicCoefficient{<:LagrangianAveraging}
const LagrangianAveragedDynamicSmagorinsky = Smagorinsky{<:Any, <:LagrangianAveragedCoefficient}

tupleit(::LagrangianAveraging) = LagrangianAveraging()
tupleit(::Colon) = Colon()
tupleit(a::Number) = tuple(a)
tupleit(a::Tuple) = a

"""
    DynamicCoefficient([FT=Float64;] averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)

When used with `Smagorinsky`, it calculates the Smagorinsky coefficient dynamically from the flow
according to the scale invariant procedure described by [Bou-Zeid et al. (2005)](@cite BouZeid05).

`DynamicCoefficient` requires an `averaging` procedure, which can be a `LagrangianAveraging` (which
averages fluid parcels along their Lagrangian trajectory) or a tuple of integers indicating
a directional averaging procedure along chosen dimensions (e.g. `averaging=(1, 2)` uses averages
in the `x` and `y` directions).

`DynamicCoefficient` is updated according to `schedule`, and `minimum_numerator` defines the minimum
value that is acceptable in the denominator of the final calculation.

Examples
========

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> dynamic_coeff = DynamicCoefficient(averaging=(1, 2))
DynamicCoefficient with
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
└── minimum_numerator = 1.0e-32

julia> using Oceananigans.TurbulenceClosures.Smagorinskys: Smagorinsky

julia> dynamic_smagorinsky = Smagorinsky(coefficient=dynamic_coeff)
DynamicSmagorinsky{Float64}:
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

To alleviate the computational cost of the `DynamicCoefficient` calculation,
users may introduce an approximation wherein the dynamic coefficient is recomputed only
every so often. For example,

```jldoctest
julia> using Oceananigans, Oceananigans.TurbulenceClosures

julia> dynamic_coeff = DynamicCoefficient(averaging=(1, 2), schedule=IterationInterval(4))
DynamicCoefficient with
├── averaging = (1, 2)
├── schedule = IterationInterval(4, 0)
└── minimum_numerator = 1.0e-32

julia> using Oceananigans.TurbulenceClosures.Smagorinskys: Smagorinsky

julia> dynamic_smagorinsky = Smagorinsky(coefficient=dynamic_coeff)
DynamicSmagorinsky{Float64}:
├── averaging = (1, 2)
├── schedule = IterationInterval(4, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

`schedule`s other than `IterationInterval(1)` are valid if the coefficient at any particular location
changes slowly relative to the frequency of recalculation.
Some published studies compute the dynamic coefficient every 5 steps
(e.g., [Bou-Zeid et al. (2005)](@cite BouZeid05); [Chen et al. 2016](@cite Chen2016);
[Salesky et al. (2017)](@cite Salesky2017); [Chor et al. 2021](@cite Chor2021))
to balance fidelity with computational cost.

See also [`DynamicSmagorinsky`](@ref).

References
==========

Chen, Bicheng, Yang, Di, Meneveau, Charles and Chamecki, Marcelo. (2016) Effects of swell on
transport and dispersion of oil plumes within the ocean mixed layer, Journal of Geophysical
Research: Oceans, 121(5), pp.3564-3578.

Salesky, Scott T., Chamecki, Marcelo, and Bou-Zeid Elie. (2017) On the nature of the transition between
roll and cellular organization in the convective boundary layer, Boundary-layer meteorology 163, 41-68.

Chor, Tomas, McWilliams, James C., Chamecki, Marcelo. (2021) Modifications to the K-Profile
Parameterization with nondiffusive fluxes for Langmuir turbulence, Journal of Physical Oceanography,
51(5), pp.1503-1521.
"""
function DynamicCoefficient(FT=Oceananigans.defaults.FloatType; averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)
    minimum_numerator = convert(FT, minimum_numerator)
    averaging = tupleit(averaging)
    return DynamicCoefficient(averaging, minimum_numerator, schedule)
end

Base.summary(dc::DynamicCoefficient) = string("DynamicCoefficient(averaging = $(dc.averaging), schedule = $(dc.schedule))")

Base.show(io::IO, dc::DynamicCoefficient) = print(io, "DynamicCoefficient with\n",
                                                      "├── averaging = ", dc.averaging, "\n",
                                                      "├── schedule = ", dc.schedule, "\n",
                                                      "└── minimum_numerator = ", dc.minimum_numerator)

function Base.show(io::IO, c::DynamicSmagorinsky)

    FT = eltype(c.coefficient.minimum_numerator)

    print(io, "DynamicSmagorinsky{$FT}:", '\n',
              "├── averaging = ", c.coefficient.averaging, '\n',
              "├── schedule = ", c.coefficient.schedule, '\n',
              "├── Pr = ", c.Pr, '\n',
              "└── minimum_numerator = ", c.coefficient.minimum_numerator)
end



#####
##### Some common utilities independent of averaging
#####

@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::DynamicSmagorinsky, closure_fields, args...)
    𝒥ᴸᴹ = closure_fields.𝒥ᴸᴹ
    𝒥ᴹᴹ = closure_fields.𝒥ᴹᴹ
    𝒥ᴸᴹ_min = closure.coefficient.minimum_numerator

    @inbounds begin
        𝒥ᴸᴹ_ijk = max(𝒥ᴸᴹ[i, j, k], 𝒥ᴸᴹ_min)
        𝒥ᴹᴹ_ijk = 𝒥ᴹᴹ[i, j, k]
    end

    return 𝒥ᴸᴹ_ijk / 𝒥ᴹᴹ_ijk * (𝒥ᴹᴹ_ijk > 0)
end

@kernel function _compute_Σ!(Σ, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Σsq = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
        Σ[i, j, k] = sqrt(Σsq)
    end
end

@kernel function _compute_Σ̄!(Σ̄, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Σ̄sq = Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
        Σ̄[i, j, k] = sqrt(Σ̄sq)
    end
end

@kernel function _compute_LM_MM!(LM, MM, Σ, Σ̄, grid, u, v, w)
    i, j, k = @index(Global, NTuple)
    LM_ijk, MM_ijk = LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    @inbounds begin
        LM[i, j, k] = LM_ijk
        MM[i, j, k] = MM_ijk
    end
end

@inline function LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    L₁₁ = L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₂ = L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₃₃ = L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₂ = L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₃ = L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₃ = L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

    M₁₁ = M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)
    M₂₂ = M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)
    M₃₃ = M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)
    M₁₂ = M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)
    M₁₃ = M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)
    M₂₃ = M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ, Σ̄)

    LM_ijk = L₁₁ * M₁₁ + L₂₂ * M₂₂ + L₃₃ * M₃₃ + 2L₁₂ * M₁₂ + 2L₁₃ * M₁₃ + 2L₂₃ * M₂₃
    MM_ijk = M₁₁ * M₁₁ + M₂₂ * M₂₂ + M₃₃ * M₃₃ + 2M₁₂ * M₁₂ + 2M₁₃ * M₁₃ + 2M₂₃ * M₂₃

    return LM_ijk, MM_ijk
end

#####
##### Directionally-averaged functionality
#####

function compute_coefficient_fields!(closure_fields, closure::DirectionallyAveragedDynamicSmagorinsky, model; parameters)
    grid = model.grid
    arch = architecture(grid)
    velocities = model.velocities
    cˢ = closure.coefficient

    if cˢ.schedule(model)
        Σ = closure_fields.Σ
        Σ̄ = closure_fields.Σ̄
        launch!(arch, grid, :xyz, _compute_Σ!, Σ, grid, velocities...)
        launch!(arch, grid, :xyz, _compute_Σ̄!, Σ̄, grid, velocities...)

        # Fill Σ, Σ̄ halos because the M tensor computation uses `filter`
        # which reads from neighboring cells (including halo cells).
        fill_halo_regions!(Σ; only_local_halos=true)
        fill_halo_regions!(Σ̄; only_local_halos=true)

        LM = closure_fields.LM
        MM = closure_fields.MM
        launch!(arch, grid, :xyz, _compute_LM_MM!, LM, MM, Σ, Σ̄, grid, velocities...)

        𝒥ᴸᴹ = closure_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ = closure_fields.𝒥ᴹᴹ
        compute!(𝒥ᴸᴹ)
        compute!(𝒥ᴹᴹ)
    end

    return nothing
end

function allocate_coefficient_fields(closure::DirectionallyAveragedDynamicSmagorinsky, grid, clock)
    LM = CenterField(grid)
    MM = CenterField(grid)

    Σ = CenterField(grid)
    Σ̄ = CenterField(grid)

    𝒥ᴸᴹ = Field(Average(LM, dims=closure.coefficient.averaging))
    𝒥ᴹᴹ = Field(Average(MM, dims=closure.coefficient.averaging))

    return (; Σ, Σ̄, LM, MM, 𝒥ᴸᴹ, 𝒥ᴹᴹ)
end

#####
##### Lagrangian-averaged functionality
#####

const c = Center()

@inline displace_node(node, δ) = node - δ
@inline displace_node(::Nothing, δ) = zero(δ)

@kernel function _lagrangian_average_LM_MM!(𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, 𝒥ᴸᴹ_min, Σ, Σ̄, grid, Δt, u, v, w)
    i, j, k = @index(Global, NTuple)
    LM, MM = LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    FT = eltype(grid)

    @inbounds begin
        𝒥ᴸᴹ⁻ᵢⱼₖ = max(𝒥ᴸᴹ⁻[i, j, k], 𝒥ᴸᴹ_min)
        𝒥ᴹᴹ⁻ᵢⱼₖ = 𝒥ᴹᴹ⁻[i, j, k]

        # Compute time scale
        𝒥ᴸᴹ𝒥ᴹᴹ = 𝒥ᴸᴹ⁻ᵢⱼₖ * 𝒥ᴹᴹ⁻ᵢⱼₖ

        T⁻ = convert(FT, 1.5) * Δᶠ(i, j, k, grid) / ∜(∜(𝒥ᴸᴹ𝒥ᴹᴹ))
        τ = Δt / T⁻
        ϵ = τ / (1 + τ)

        # Compute interpolation
        x = xnode(i, j, k, grid, c, c, c)
        y = ynode(i, j, k, grid, c, c, c)
        z = znode(i, j, k, grid, c, c, c)

        # Displacements
        δx = u[i, j, k] * Δt
        δy = v[i, j, k] * Δt
        δz = w[i, j, k] * Δt
        # Prevent displacements from getting too big?
        Δx = Δxᶜᶜᶜ(i, j, k, grid)
        Δy = Δyᶜᶜᶜ(i, j, k, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)

        δx = clamp(δx, -Δx, Δx)
        δy = clamp(δy, -Δy, Δy)
        δz = clamp(δz, -Δz, Δz)

        # Previous locations
        x⁻ = displace_node(x, δx)
        y⁻ = displace_node(y, δy)
        z⁻ = displace_node(z, δz)
        X⁻ = (x⁻, y⁻, z⁻)

        itp_𝒥ᴹᴹ⁻ = interpolate(X⁻, 𝒥ᴹᴹ⁻, (c, c, c), grid)
        itp_𝒥ᴸᴹ⁻ = interpolate(X⁻, 𝒥ᴸᴹ⁻, (c, c, c), grid)

        # Take time-step
        𝒥ᴹᴹ[i, j, k] = ϵ * MM + (1 - ϵ) * itp_𝒥ᴹᴹ⁻

        𝒥ᴸᴹ★ = ϵ * LM + (1 - ϵ) * max(itp_𝒥ᴸᴹ⁻, 𝒥ᴸᴹ_min)
        𝒥ᴸᴹ[i, j, k] = max(𝒥ᴸᴹ★, 𝒥ᴸᴹ_min)
    end
end

function initialize_closure_fields!(closure_fields, closure::LagrangianAveragedDynamicSmagorinsky, model)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    cˢ = closure.coefficient
    u, v, w = model.velocities

    Σ = closure_fields.Σ
    Σ̄ = closure_fields.Σ̄
    launch!(arch, grid, :xyz, _compute_Σ!, Σ, grid, u, v, w)
    launch!(arch, grid, :xyz, _compute_Σ̄!, Σ̄, grid, u, v, w)

    # Fill Σ, Σ̄ halos because the M tensor computation uses `filter`
    # which reads from neighboring cells (including halo cells).
    fill_halo_regions!(Σ; only_local_halos=true)
    fill_halo_regions!(Σ̄; only_local_halos=true)

    𝒥ᴸᴹ  = closure_fields.𝒥ᴸᴹ
    𝒥ᴹᴹ  = closure_fields.𝒥ᴹᴹ
    𝒥ᴸᴹ_min = cˢ.minimum_numerator

    # Compute instantaneous LM, MM and spatially average for initialization.
    # We clip LM to zero before averaging because the spatial mean of LM can be negative
    # for random or weakly-turbulent velocity fields, which would set 𝒥ᴸᴹ to minimum_numerator
    # and cause the Lagrangian time scale to diverge, preventing any evolution.
    launch!(arch, grid, :xyz, _compute_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, Σ, Σ̄, grid, u, v, w)
    parent(𝒥ᴸᴹ) .= max(mean(x -> max(x, zero(𝒥ᴸᴹ_min)), 𝒥ᴸᴹ), 𝒥ᴸᴹ_min)
    parent(𝒥ᴹᴹ) .= mean(𝒥ᴹᴹ)

    # Initialize previous_compute_time to current time
    closure_fields.previous_compute_time[] = clock.time

    return nothing
end

function step_closure_prognostics!(closure_fields, closure::LagrangianAveragedDynamicSmagorinsky, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    cˢ = closure.coefficient
    u, v, w = model.velocities

    time_to_compute = isnothing(cˢ.schedule) || (cˢ.schedule(model) && clock.stage == 1)

    if time_to_compute
        Σ = closure_fields.Σ
        Σ̄ = closure_fields.Σ̄
        launch!(arch, grid, :xyz, _compute_Σ!, Σ, grid, u, v, w)
        launch!(arch, grid, :xyz, _compute_Σ̄!, Σ̄, grid, u, v, w)

        # Fill Σ, Σ̄ halos because the M tensor computation uses `filter`
        # which reads from neighboring cells (including halo cells).
        fill_halo_regions!(Σ; only_local_halos=true)
        fill_halo_regions!(Σ̄; only_local_halos=true)

        parent(closure_fields.𝒥ᴸᴹ⁻) .= parent(closure_fields.𝒥ᴸᴹ)
        parent(closure_fields.𝒥ᴹᴹ⁻) .= parent(closure_fields.𝒥ᴹᴹ)

        𝒥ᴸᴹ⁻ = closure_fields.𝒥ᴸᴹ⁻
        𝒥ᴹᴹ⁻ = closure_fields.𝒥ᴹᴹ⁻
        𝒥ᴸᴹ  = closure_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ  = closure_fields.𝒥ᴹᴹ
        𝒥ᴸᴹ_min = cˢ.minimum_numerator

        # Compute time elapsed since last coefficient computation
        Δt_lagrangian = clock.time - closure_fields.previous_compute_time[]
        closure_fields.previous_compute_time[] = clock.time

        launch!(arch, grid, :xyz,
                _lagrangian_average_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, 𝒥ᴸᴹ_min, Σ, Σ̄, grid, Δt_lagrangian, u, v, w)
    end

    return nothing
end

# Lagrangian-averaged coefficients are now stepped via step_closure_prognostics!
compute_coefficient_fields!(closure_fields, closure::LagrangianAveragedDynamicSmagorinsky, model; parameters) = nothing

function allocate_coefficient_fields(closure::LagrangianAveragedDynamicSmagorinsky, grid, clock)
    𝒥ᴸᴹ⁻ = CenterField(grid)
    𝒥ᴹᴹ⁻ = CenterField(grid)

    𝒥ᴸᴹ = CenterField(grid)
    𝒥ᴹᴹ = CenterField(grid)

    Σ = CenterField(grid)
    Σ̄ = CenterField(grid)

    # Initialize to clock.time; will be properly set during initialize_closure_fields!
    previous_compute_time = Ref(clock.time)

    return (; Σ, Σ̄, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, previous_compute_time)
end

#####
##### Checkpointing
#####

const DirectionallyAveragedSmagorinskyFields = NamedTuple{(:νₑ, :Σ, :Σ̄, :LM, :MM, :𝒥ᴸᴹ, :𝒥ᴹᴹ)}
const LagrangianAveragedSmagorinskyFields = NamedTuple{(:νₑ, :Σ, :Σ̄, :𝒥ᴸᴹ, :𝒥ᴹᴹ, :𝒥ᴸᴹ⁻, :𝒥ᴹᴹ⁻, :previous_compute_time)}

function prognostic_state(cf::DirectionallyAveragedSmagorinskyFields)
    return (𝒥ᴸᴹ = prognostic_state(cf.𝒥ᴸᴹ),
            𝒥ᴹᴹ = prognostic_state(cf.𝒥ᴹᴹ))
end

function restore_prognostic_state!(restored::DirectionallyAveragedSmagorinskyFields, from)
    restore_prognostic_state!(restored.𝒥ᴸᴹ, from.𝒥ᴸᴹ)
    restore_prognostic_state!(restored.𝒥ᴹᴹ, from.𝒥ᴹᴹ)
    return restored
end

function prognostic_state(cf::LagrangianAveragedSmagorinskyFields)
    return (𝒥ᴸᴹ = prognostic_state(cf.𝒥ᴸᴹ),
            𝒥ᴹᴹ = prognostic_state(cf.𝒥ᴹᴹ),
            𝒥ᴸᴹ⁻ = prognostic_state(cf.𝒥ᴸᴹ⁻),
            𝒥ᴹᴹ⁻ = prognostic_state(cf.𝒥ᴹᴹ⁻),
            previous_compute_time = cf.previous_compute_time[])
end

function restore_prognostic_state!(restored::LagrangianAveragedSmagorinskyFields, from)
    restore_prognostic_state!(restored.𝒥ᴸᴹ, from.𝒥ᴸᴹ)
    restore_prognostic_state!(restored.𝒥ᴹᴹ, from.𝒥ᴹᴹ)
    restore_prognostic_state!(restored.𝒥ᴸᴹ⁻, from.𝒥ᴸᴹ⁻)
    restore_prognostic_state!(restored.𝒥ᴹᴹ⁻, from.𝒥ᴹᴹ⁻)
    restored.previous_compute_time[] = from.previous_compute_time
    return restored
end

restore_prognostic_state!(::DirectionallyAveragedSmagorinskyFields, ::Nothing) = nothing
restore_prognostic_state!(::LagrangianAveragedSmagorinskyFields, ::Nothing) = nothing
