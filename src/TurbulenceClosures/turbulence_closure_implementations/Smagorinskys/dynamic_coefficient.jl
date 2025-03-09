using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interpolate
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Statistics

struct DynamicCoefficient{A, FT, S}
    averaging :: A
    minimum_numerator :: FT
    schedule :: S
end

const DynamicSmagorinsky = Smagorinsky{<:Any, <:DynamicCoefficient}

function DynamicSmagorinsky(time_discretization=ExplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType; averaging,
                            Pr=1.0, schedule=IterationInterval(1), minimum_numerator=1e-32)
    coefficient = DynamicCoefficient(FT; averaging, schedule, minimum_numerator)
    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return Smagorinsky{TD}(coefficient, Pr)
end

DynamicSmagorinsky(FT::DataType; kwargs...) = DynamicSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

Adapt.adapt_structure(to, dc::DynamicCoefficient) = DynamicCoefficient(dc.averaging, dc.minimum_numerator, nothing)

const DirectionallyAveragedCoefficient{N} = DynamicCoefficient{<:Union{NTuple{N, Int}, Int, Colon}} where N
const DirectionallyAveragedDynamicSmagorinsky{N} = Smagorinsky{<:Any, <:DirectionallyAveragedCoefficient{N}} where N

struct LagrangianAveraging end
const LagrangianAveragedCoefficient = DynamicCoefficient{<:LagrangianAveraging}
const LagrangianAveragedDynamicSmagorinsky = Smagorinsky{<:Any, <:LagrangianAveragedCoefficient}

"""
    DynamicCoefficient([FT=Float64;] averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)

When used with `Smagorinsky`, it calculates the Smagorinsky coefficient dynamically from the flow
according to the scale invariant procedure in [BouZeid05](@citet).

`DynamicCoefficient` requires an `averaging` procedure, which can be a `LagrangianAveraging` (which
averages fluid parcels along their Lagrangian trajectory) or a tuple of integers indicating
a directional averaging procedure along chosen dimensions (e.g. `averaging=(1,2)` uses averages
in the `x` and `y` directions).

`DynamicCoefficient` is updated according to `schedule`, and `minimum_numerator` defines the minimum
value that is acceptable in the denominator of the final calculation.

Examples
========

```jldoctest
julia> using Oceananigans

julia> dynamic_coeff = DynamicCoefficient(averaging=(1, 2))
DynamicCoefficient with
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
└── minimum_numerator = 1.0e-32

julia> dynamic_smagorinsky = Smagorinsky(coefficient=dynamic_coeff)
Smagorinsky closure with
├── coefficient = DynamicCoefficient(averaging = (1, 2), schedule = IterationInterval(1, 0))
└── Pr = 1.0
```

The dynamic Smagorinsky above has its coefficient recalculated at every time step, which will almost
certainly be very slow. To alleviate the high computational cost of the `DynamicCoefficient`
calculation, users may introduce an approximation wherein the dynamic coefficient is recomputed only
every so often. This is standard practice in the literature and, while in principle any frequency
choice is possible (as long as the coefficient changes relatively slowly compared to a single
time-step), all published studies seem to recalculate it every 5 steps (e.g., Bou-Zeid et al. 2005;
Chen et al. 2016; Salesky et al. 2017; Chor et al 2021). This choice seems to stem from the
results by Bou-Zeid et al. (2005) who found that considerably speed up simulations while still
producing very similar results to an update frequency of every time step. Users can change the
update frequency using the `schedule` keyword argument. For example, a `DynamicCoefficient` that
gets updated every 4 timesteps is obtained via:

```jldoctest
julia> using Oceananigans

julia> dynamic_coeff = DynamicCoefficient(averaging=(1, 2), schedule=IterationInterval(4))
DynamicCoefficient with
├── averaging = (1, 2)
├── schedule = IterationInterval(4, 0)
└── minimum_numerator = 1.0e-32

julia> dynamic_smagorinsky = Smagorinsky(coefficient=dynamic_coeff)
Smagorinsky closure with
├── coefficient = DynamicCoefficient(averaging = (1, 2), schedule = IterationInterval(4, 0))
└── Pr = 1.0
```

References
==========

Bou-Zeid, Elie, Meneveau, Charles, and Parlange, Marc. (2005) A scale-dependent Lagrangian dynamic model for
large eddy simulation of complex turbulent flows, Physics of Fluids, **17**, 025105.

Salesky, Scott T., Chamecki, Marcelo, and Bou-Zeid Elie. (2017) On the nature of the transition between
roll and cellular organization in the convective boundary layer, Boundary-layer meteorology 163, 41-68.

Chen, Bicheng, Yang, Di, Meneveau, Charles and Chamecki, Marcelo. (2016) Effects of swell on
transport and dispersion of oil plumes within the ocean mixed layer, Journal of Geophysical
Research: Oceans, 121(5), pp.3564-3578.

Chor, Tomas, McWilliams, James C., Chamecki, Marcelo. (2021) Modifications to the K-Profile
Parameterization with nondiffusive fluxes for Langmuir turbulence, Journal of Physical Oceanography,
51(5), pp.1503-1521.
"""
function DynamicCoefficient(FT=Oceananigans.defaults.FloatType; averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)
    minimum_numerator = convert(FT, minimum_numerator)
    return DynamicCoefficient(averaging, minimum_numerator, schedule)
end

Base.summary(dc::DynamicCoefficient) = string("DynamicCoefficient(averaging = $(dc.averaging), schedule = $(dc.schedule))")
Base.show(io::IO, dc::DynamicCoefficient) = print(io, "DynamicCoefficient with\n",
                                                      "├── averaging = ", dc.averaging, "\n",
                                                      "├── schedule = ", dc.schedule, "\n",
                                                      "└── minimum_numerator = ", dc.minimum_numerator)

#####
##### Some common utilities independent of averaging
#####

@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::DynamicSmagorinsky, diffusivity_fields, args...)
    𝒥ᴸᴹ = diffusivity_fields.𝒥ᴸᴹ
    𝒥ᴹᴹ = diffusivity_fields.𝒥ᴹᴹ
    𝒥ᴸᴹ_min = closure.coefficient.minimum_numerator

    @inbounds begin
        𝒥ᴸᴹ_ijk = max(𝒥ᴸᴹ[i, j, k], 𝒥ᴸᴹ_min)
        𝒥ᴹᴹ_ijk = 𝒥ᴹᴹ[i, j, k]
    end

    return ifelse(𝒥ᴹᴹ_ijk == 0, zero(grid), 𝒥ᴸᴹ_ijk / 𝒥ᴹᴹ_ijk)
end

@kernel function _compute_Σ_Σ̄!(Σ, Σ̄, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Σ[i, j, k] = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
        Σ̄[i, j, k] = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
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

function compute_coefficient_fields!(diffusivity_fields, closure::DirectionallyAveragedDynamicSmagorinsky, model; parameters)
    grid = model.grid
    arch = architecture(grid)
    velocities = model.velocities
    cˢ = closure.coefficient

    if cˢ.schedule(model)
        Σ = diffusivity_fields.Σ
        Σ̄ = diffusivity_fields.Σ̄
        launch!(arch, grid, :xyz, _compute_Σ_Σ̄!, Σ, Σ̄, grid, velocities...)

        LM = diffusivity_fields.LM
        MM = diffusivity_fields.MM
        launch!(arch, grid, :xyz, _compute_LM_MM!, LM, MM, Σ, Σ̄, grid, velocities...)

        𝒥ᴸᴹ = diffusivity_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ = diffusivity_fields.𝒥ᴹᴹ
        compute!(𝒥ᴸᴹ)
        compute!(𝒥ᴹᴹ)
    end

    return nothing
end

function allocate_coefficient_fields(closure::DirectionallyAveragedDynamicSmagorinsky, grid)
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

function compute_coefficient_fields!(diffusivity_fields, closure::LagrangianAveragedDynamicSmagorinsky, model; parameters)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    cˢ = closure.coefficient
    t⁻ = diffusivity_fields.previous_compute_time
    u, v, w = model.velocities

    Δt = clock.time - t⁻[]
    t⁻[] = model.clock.time

    if cˢ.schedule(model)
        Σ = diffusivity_fields.Σ
        Σ̄ = diffusivity_fields.Σ̄
        launch!(arch, grid, :xyz, _compute_Σ_Σ̄!, Σ, Σ̄, grid, u, v, w)

        parent(diffusivity_fields.𝒥ᴸᴹ⁻) .= parent(diffusivity_fields.𝒥ᴸᴹ)
        parent(diffusivity_fields.𝒥ᴹᴹ⁻) .= parent(diffusivity_fields.𝒥ᴹᴹ)

        𝒥ᴸᴹ⁻ = diffusivity_fields.𝒥ᴸᴹ⁻
        𝒥ᴹᴹ⁻ = diffusivity_fields.𝒥ᴹᴹ⁻
        𝒥ᴸᴹ  = diffusivity_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ  = diffusivity_fields.𝒥ᴹᴹ
        𝒥ᴸᴹ_min = cˢ.minimum_numerator

        if !isfinite(clock.last_Δt) || Δt == 0 # first time-step
            launch!(arch, grid, :xyz, _compute_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, Σ, Σ̄, grid, u, v, w)
            parent(𝒥ᴸᴹ) .= max(mean(𝒥ᴸᴹ), 𝒥ᴸᴹ_min)
            parent(𝒥ᴹᴹ) .= mean(𝒥ᴹᴹ)
        else
            fill_halo_regions!(𝒥ᴹᴹ⁻)
            fill_halo_regions!(𝒥ᴸᴹ⁻)
            launch!(arch, grid, :xyz,
                    _lagrangian_average_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, 𝒥ᴸᴹ_min, Σ, Σ̄, grid, Δt, u, v, w)

        end
    end

    return nothing
end

function allocate_coefficient_fields(closure::LagrangianAveragedDynamicSmagorinsky, grid)
    𝒥ᴸᴹ⁻ = CenterField(grid)
    𝒥ᴹᴹ⁻ = CenterField(grid)

    𝒥ᴸᴹ = CenterField(grid)
    𝒥ᴹᴹ = CenterField(grid)

    Σ = CenterField(grid)
    Σ̄ = CenterField(grid)

    previous_compute_time = Ref(zero(grid))

    return (; Σ, Σ̄, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, previous_compute_time)
end


