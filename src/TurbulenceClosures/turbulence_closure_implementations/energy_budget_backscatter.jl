using Oceananigans.Grids: static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ

struct EnergyBudgetBackscatter{FT, V} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, HorizontalFormulation, 2}
    backscatter_coefficient :: FT
    subgrid_energy_diffusivity :: FT
    vertical_structure :: V
end

"""
    EnergyBudgetBackscatter(FT = Oceananigans.defaults.FloatType;
                            backscatter_coefficient = 0.4,
                            subgrid_energy_diffusivity = 600,
                            vertical_structure = 1)

Return the energy budget-based backscatter closure of [Jansen2015backscatter](@citet).
The closure applies a negative horizontal Laplacian viscosity

```
ν = - c Δ √(2 max(e, 0)) φ,
```

fed by a prognostic, vertically-averaged sub-grid eddy kinetic energy `e` that obeys

```
∂t e = - (1/H) ∫ ∑ᵢⱼ τᵢⱼ ∂ⱼuᵢ dz + (1/H) ∇ₕ⋅(H κₑ ∇ₕe),
```

where `Δ = √(Δx Δy)`, `H` is the column depth, and the sum runs over the horizontal viscous fluxes
`τᵢⱼ`, which force momentum as `-∂ⱼτᵢⱼ`, of every closure in the model's closure tuple. Energy removed from the resolved flow by
horizontal viscosity therefore enters `e`, and the backscatter returns it to the resolved flow.
Vertical viscosities contribute nothing to the sum.

Keyword arguments
=================

- `backscatter_coefficient`: Nondimensional coefficient `c`.
- `subgrid_energy_diffusivity`: Horizontal diffusivity `κₑ` of sub-grid eddy kinetic energy (m² s⁻¹).
- `vertical_structure`: Nondimensional vertical structure `φ` of the backscatter viscosity:
  a `Number`, an `AbstractArray` or `Field` at cell centers, or a function of `(x, y, z)`.

Example
=======

```jldoctest
using Oceananigans

closure = EnergyBudgetBackscatter()

# output
EnergyBudgetBackscatter(backscatter_coefficient=0.4, subgrid_energy_diffusivity=600.0, vertical_structure=1)
```
"""
function EnergyBudgetBackscatter(FT = Oceananigans.defaults.FloatType;
                                 backscatter_coefficient = 0.4,
                                 subgrid_energy_diffusivity = 600,
                                 vertical_structure = 1)

    return EnergyBudgetBackscatter(convert(FT, backscatter_coefficient),
                                   convert(FT, subgrid_energy_diffusivity),
                                   vertical_structure)
end

Adapt.adapt_structure(to, closure::EnergyBudgetBackscatter) =
    EnergyBudgetBackscatter(closure.backscatter_coefficient,
                            closure.subgrid_energy_diffusivity,
                            adapt(to, closure.vertical_structure))

@inline viscosity(::EnergyBudgetBackscatter, K) = K.νₑ

for flux in (:diffusive_flux_x, :diffusive_flux_y, :viscous_flux_wx, :viscous_flux_wy)
    @eval @inline $flux(i, j, k, grid, ::EnergyBudgetBackscatter, args...) = zero(grid)
end

function build_closure_fields(grid, clock, tracer_names, bcs, ::EnergyBudgetBackscatter)
    e = Field{Center, Center, Nothing}(grid)
    Ge = Field{Center, Center, Nothing}(grid)
    νₑ = CenterField(grid)
    return (; e, Ge, νₑ)
end

@kernel function _compute_backscatter_viscosity!(νₑ, grid, closure, e)
    i, j, k = @index(Global, NTuple)

    φ = closure_coefficient(i, j, k, grid, closure.vertical_structure)
    Δ = sqrt(Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid))
    e⁺ = @inbounds max(e[i, j, 1], zero(grid))

    @inbounds νₑ[i, j, k] = - closure.backscatter_coefficient * Δ * sqrt(2e⁺) * φ
end

function compute_closure_fields!(closure_fields, closure::EnergyBudgetBackscatter, model; parameters = :xyz)
    launch!(model.architecture, model.grid, parameters, _compute_backscatter_viscosity!,
            closure_fields.νₑ, model.grid, closure, closure_fields.e)

    return nothing
end

# Volume-integrated tendency of resolved kinetic energy, ∑ᵢⱼ τᵢⱼ ∂ⱼuᵢ, at the locations of τᵢⱼ
@inline viscous_energy_transferᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    Ax_qᶜᶜᶜ(i, j, k, grid, _viscous_flux_ux, closure, K, clock, fields, b) * δxᶜᵃᵃ(i, j, k, grid, fields.u) +
    Ay_qᶜᶜᶜ(i, j, k, grid, _viscous_flux_vy, closure, K, clock, fields, b) * δyᵃᶜᵃ(i, j, k, grid, fields.v)

@inline viscous_energy_transferᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    Ay_qᶠᶠᶜ(i, j, k, grid, _viscous_flux_uy, closure, K, clock, fields, b) * δyᵃᶠᵃ(i, j, k, grid, fields.u) +
    Ax_qᶠᶠᶜ(i, j, k, grid, _viscous_flux_vx, closure, K, clock, fields, b) * δxᶠᵃᵃ(i, j, k, grid, fields.v)

@inline subgrid_energy_fluxᶠᶜᶜ(i, j, k, grid, κₑ, e) = κₑ * static_column_depthᶠᶜᵃ(i, j, grid) * Δy_qᶠᶜᶜ(i, j, k, grid, ∂xᵣᶠᶜᶜ, e)
@inline subgrid_energy_fluxᶜᶠᶜ(i, j, k, grid, κₑ, e) = κₑ * static_column_depthᶜᶠᵃ(i, j, grid) * Δx_qᶜᶠᶜ(i, j, k, grid, ∂yᵣᶜᶠᶜ, e)

@kernel function _compute_subgrid_energy_tendency!(Ge, grid, e, κₑ, closure, K, clock, fields, b)
    i, j = @index(Global, NTuple)

    ε = zero(grid)
    for k in 1:grid.Nz
        ε += viscous_energy_transferᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) +
             ℑxyᶜᶜᵃ(i, j, k, grid, viscous_energy_transferᶠᶠᶜ, closure, K, clock, fields, b)
    end

    ∇F = δxᶜᵃᵃ(i, j, 1, grid, subgrid_energy_fluxᶠᶜᶜ, κₑ, e) +
         δyᵃᶜᵃ(i, j, 1, grid, subgrid_energy_fluxᶜᶠᶜ, κₑ, e)

    H = static_column_depthᶜᶜᵃ(i, j, grid)

    @inbounds Ge[i, j, 1] = ifelse(H > 0, Az⁻¹ᶜᶜᶜ(i, j, 1, grid) * (∇F - ε) / H, zero(grid))
end

function step_closure_prognostics!(closure_fields, closure::EnergyBudgetBackscatter, model, Δt)
    e = closure_fields.e
    Ge = closure_fields.Ge

    launch!(model.architecture, model.grid, :xy, _compute_subgrid_energy_tendency!,
            Ge, model.grid, e, closure.subgrid_energy_diffusivity,
            model.closure, model.closure_fields, model.clock, fields(model), buoyancy_force(model))

    parent(e) .+= Δt .* parent(Ge)
    fill_halo_regions!(e)

    return nothing
end

Base.summary(closure::EnergyBudgetBackscatter) =
    string("EnergyBudgetBackscatter(backscatter_coefficient=", prettysummary(closure.backscatter_coefficient),
           ", subgrid_energy_diffusivity=", prettysummary(closure.subgrid_energy_diffusivity),
           ", vertical_structure=", prettysummary(closure.vertical_structure), ")")

Base.show(io::IO, closure::EnergyBudgetBackscatter) = print(io, summary(closure))
