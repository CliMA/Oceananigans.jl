"""
    HorizontalCurvilinearDiffusivity{N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct HorizontalCurvilinearDiffusivity{N, K} <: AbstractTurbulenceClosure
    ν :: N
    κ :: K
end

"""
    HorizontalCurvilinearDiffusivity(; ν=0, κ=0)

Returns parameters for an horizontal diffusivity model on
curvilinear grids with viscosity `ν` and diffusivities `κ` for each tracer
field in `tracers`. `ν` and the fields of `κ` may be constants or functions
of `(x, y, z, t)`, and may represent molecular diffusivities in cases that all flow
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of
unresolved, subgrid-scale turbulence.
`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
"""
function HorizontalCurvilinearDiffusivity(FT=Float64; ν=ν₀, κ=κ₀)
    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return HorizontalCurvilinearDiffusivity(FT(ν), κ)
    else
        return HorizontalCurvilinearDiffusivity(ν, κ)
    end
end

function with_tracers(tracers, closure::HorizontalCurvilinearDiffusivity)
    κ = tracer_diffusivities(tracers, closure.κ)
    return HorizontalCurvilinearDiffusivity(closure.ν, κ)
end

calculate_diffusivities!(K, arch, grid, closure::HorizontalCurvilinearDiffusivity, args...) = nothing

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν) = ν
@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Center, grid, i), ynode(Center, grid, j), znode(Center, grid, k), clock.time)

@inline νᶠᶠᶜ(i, j, k, grid, clock, ν) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face, grid, i), ynode(Face, grid, j), znode(Cell, grid, k), clock.time)

@inline ν_δᶜᶜᵃ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᵃ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᵃ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)
    
@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::HorizontalCurvilinearDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κ = closure.κ[tracer_index]

    return ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κ, κ, 0, c)
end

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::HorizontalCurvilinearDiffusivity, U, args...) = (
      δxᶠᵃᵃ(i, j, k, grid, ν_δᶜᶜᵃ, clock, closure.ν, U.u, U.v) / Δxᶠᶜᵃ(i, j, k, grid)
    - δyᵃᶜᵃ(i, j, k, grid, ν_ζᶠᶠᵃ, clock, closure.ν, U.u, U.v) / Δyᶠᶜᵃ(i, j, k, grid)
)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::HorizontalCurvilinearDiffusivity, U, args...) = (
      δyᵃᶠᵃ(i, j, k, grid, ν_δᶜᶜᵃ, clock, closure.ν, U.u, U.v) / Δyᶜᶠᵃ(i, j, k, grid)
    + δxᶜᵃᵃ(i, j, k, grid, ν_ζᶠᶠᵃ, clock, closure.ν, U.u, U.v) / Δxᶜᶠᵃ(i, j, k, grid)
)

Base.show(io::IO, closure::HorizontalCurvilinearDiffusivity) =
    print(io, "HorizontalCurvilinearDiffusivity: ν=$(closure.ν), κ=$(closure.κ)")
