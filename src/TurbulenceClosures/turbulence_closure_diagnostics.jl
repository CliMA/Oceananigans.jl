using Oceananigans.Diagnostics: Diagnostics, cell_diffusion_timescale
using Oceananigans.Grids: minimum_xspacing, minimum_yspacing, minimum_zspacing

#####
##### Timescale for diffusion across one cell
#####

function min_Δxyz(grid, ::ThreeDimensionalFormulation)
    Δx = minimum_xspacing(grid, Center(), Center(), Center())
    Δy = minimum_yspacing(grid, Center(), Center(), Center())
    Δz = minimum_zspacing(grid, Center(), Center(), Center())
    return min(Δx, Δy, Δz)
end

function min_Δxyz(grid, ::HorizontalFormulation)
    Δx = minimum_xspacing(grid, Center(), Center(), Center())
    Δy = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx, Δy)
end

min_Δxyz(grid, ::VerticalFormulation) = minimum_zspacing(grid, Center(), Center(), Center())


Diagnostics.cell_diffusion_timescale(model) = cell_diffusion_timescale(model.closure, model.closure_fields,
                                                                       model.grid, model.clock, fields(model))
Diagnostics.cell_diffusion_timescale(::Nothing, closure_fields, grid, clock, fields) = Inf

maximum_numeric_diffusivity(κ::Number, grid, clock, fields) = κ
maximum_numeric_diffusivity(κ::FunctionField, grid, clock, fields) = maximum(κ)
maximum_numeric_diffusivity(κ_tuple::NamedTuple, grid, clock, fields) = maximum(maximum_numeric_diffusivity(κ, grid, clock, fields) for κ in κ_tuple)
maximum_numeric_diffusivity(κ::NamedTuple{()}, grid, clock, fields) = 0 # tracers=nothing means empty diffusivity tuples
maximum_numeric_diffusivity(::Nothing, grid, clock, fields) = 0

const FunctionDiffusivity = Union{<:DiscreteDiffusionFunction, <:Function}

function maximum_numeric_diffusivity(κ::FunctionDiffusivity, grid, clock, fields)

    location = (Center(), Center(), Center())
    diffusivity_kfo = KernelFunctionOperation{Center, Center, Center}(κᶜᶜᶜ, grid, location, κ, clock, fields)
    return maximum(diffusivity_kfo)
end

function Diagnostics.cell_diffusion_timescale(closure::ScalarDiffusivity{TD, Dir}, closure_fields, grid, clock, fields) where {TD, Dir}
    Δ = min_Δxyz(grid, formulation(closure))
    max_κ = maximum_numeric_diffusivity(closure.κ, grid, clock, fields)
    max_ν = maximum_numeric_diffusivity(closure.ν, grid, clock, fields)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

function Diagnostics.cell_diffusion_timescale(closure::ScalarBiharmonicDiffusivity{Dir}, closure_fields, grid, clock, fields) where {Dir}
    Δ = min_Δxyz(grid, formulation(closure))
    max_κ = maximum_numeric_diffusivity(closure.κ, grid, clock, fields)
    max_ν = maximum_numeric_diffusivity(closure.ν, grid, clock, fields)
    return min(Δ^4/ max_ν, Δ^4 / max_κ)
end

function Diagnostics.cell_diffusion_timescale(closure::Smagorinsky, closure_fields, grid, clock, fields)
    Δ = min_Δxyz(grid, formulation(closure))
    min_Pr = closure.Pr isa NamedTuple{()} ? 1 : minimum(closure.Pr) # Innocuous value is there's no tracers
    max_νκ = maximum(closure_fields.νₑ.data.parent) * max(1, 1/min_Pr)
    return Δ^2 / max_νκ
end

function Diagnostics.cell_diffusion_timescale(closure::AnisotropicMinimumDissipation, closure_fields, grid, clock, fields)
    Δ = min_Δxyz(grid, formulation(closure))
    max_ν = maximum(closure_fields.νₑ.data.parent)
    max_κ = closure_fields.κₑ isa NamedTuple{()} ? Inf : max(Tuple(maximum(κₑ.data.parent) for κₑ in closure_fields.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

function Diagnostics.cell_diffusion_timescale(closure::TwoDimensionalLeith, closure_fields, grid, clock, fields)
    Δ = min_Δxyz(grid, ThreeDimensionalFormulation())
    max_ν = maximum(closure_fields.νₑ.data.parent)
    return Δ^2 / max_ν
end

# Vertically-implicit treatment of vertical diffusivity has no time-step restriction
Diagnostics.cell_diffusion_timescale(::ConvectiveAdjustmentVerticalDiffusivity{<:VerticallyImplicitTimeDiscretization},
                                     closure_fields, grid, clock, fields) = Inf

Diagnostics.cell_diffusion_timescale(closure::Tuple, closure_fields, grid, clock, fields) =
    minimum(cell_diffusion_timescale(c, cf, grid, clock, fields) for (c, cf) in zip(closure, closure_fields))
