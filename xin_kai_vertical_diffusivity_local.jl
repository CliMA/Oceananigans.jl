using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.Grids: inactive_node
using Oceananigans.Operators: ℑzᵃᵃᶜ, ℑxyᶠᶠᵃ, ℑxyᶜᶜᵃ

using Adapt

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.TurbulenceClosures:
        tapering_factorᶠᶜᶜ,
        tapering_factorᶜᶠᶜ,
        tapering_factorᶜᶜᶠ,
        tapering_factor,
        SmallSlopeIsopycnalTensor,
        AbstractScalarDiffusivity,
        ExplicitTimeDiscretization,
        FluxTapering,
        isopycnal_rotation_tensor_xz_ccf,
        isopycnal_rotation_tensor_yz_ccf,
        isopycnal_rotation_tensor_zz_ccf

import Oceananigans.TurbulenceClosures:
        compute_diffusivities!,
        DiffusivityFields,
        viscosity, 
        diffusivity,
        getclosure,
        top_buoyancy_flux,
        diffusive_flux_x,
        diffusive_flux_y, 
        diffusive_flux_z,
        viscous_flux_ux,
        viscous_flux_vx,
        viscous_flux_uy,
        viscous_flux_vy

using Oceananigans.Utils: launch!
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b 

using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation, AbstractScalarDiffusivity
using Oceananigans.TurbulenceClosures: AbstractScalarBiharmonicDiffusivity
using Oceananigans.Operators
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, ℑxyᶜᶜᵃ, ζ₃ᶠᶠᶜ, div_xyᶜᶜᶜ
using Oceananigans.Operators: Δx, Δy
using Oceananigans.Operators: ℑxyz

using Oceananigans.Operators: ℑxyzᶜᶜᶠ, ℑyzᵃᶜᶠ, ℑxzᶜᵃᶠ, Δxᶜᶜᶜ, Δyᶜᶜᶜ

using Oceananigans.BoundaryConditions

struct XinKaiLocalVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    ν₀  :: FT
    νˢʰ :: FT
    νᶜⁿ :: FT
    Pr_convₜ :: FT
    Pr_shearₜ :: FT
    Riᶜ :: FT
    δRi :: FT
end

function XinKaiLocalVerticalDiffusivity{TD}(ν₀  :: FT, 
                                            νˢʰ :: FT,
                                            νᶜⁿ :: FT,
                                            Pr_convₜ :: FT,
                                            Pr_shearₜ :: FT,
                                            Riᶜ :: FT,
				                            δRi :: FT) where {TD, FT}
                                       
    return XinKaiLocalVerticalDiffusivity{TD, FT}(ν₀, νˢʰ, νᶜⁿ, Pr_convₜ, Pr_shearₜ, Riᶜ, δRi)
end

function XinKaiLocalVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                        FT  = Float64;
                                        ν₀  = 1e-5, 
                                        νˢʰ = 0.04569735882746968,
                                        νᶜⁿ = 0.47887785611155065,
                                        Pr_convₜ = 0.1261854430705509,
                                        Pr_shearₜ = 1.594794053970444,
                                        Riᶜ = 0.9964350402840053,
                                        δRi = 0.05635304878092709) 

    TD = typeof(time_discretization)

    return XinKaiLocalVerticalDiffusivity{TD}(convert(FT, ν₀),
                                              convert(FT, νˢʰ),
                                              convert(FT, νᶜⁿ),
                                              convert(FT, Pr_convₜ),
                                              convert(FT, Pr_shearₜ),
                                              convert(FT, Riᶜ),
                                              convert(FT, δRi))
end

XinKaiLocalVerticalDiffusivity(FT::DataType; kw...) =
    XinKaiLocalVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::XinKaiLocalVerticalDiffusivity{TD, FT}) where {TD, FT} = 
    XinKaiLocalVerticalDiffusivity{TD, FT}(clo.ν₀, clo.νˢʰ, clo.νᶜⁿ, clo.Pr_convₜ, clo.Pr_shearₜ, clo.Riᶜ, clo.δRi)
                                         
#####                                    
##### Diffusivity field utilities        
#####                                    
                                         
const RBVD = XinKaiLocalVerticalDiffusivity   
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfXKVD = Union{RBVD, RBVDArray}
const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfXKVD)   = (c, c, f)
@inline diffusivity_location(::FlavorOfXKVD) = (c, c, f)

@inline viscosity(::FlavorOfXKVD, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfXKVD, diffusivities, id) = diffusivities.κᶜ

with_tracers(tracers, closure::FlavorOfXKVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfXKVD)
    κᶜ = Field((Center, Center, Face), grid)
    κᵘ = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κᶜ, κᵘ, Ri)
end

function compute_diffusivities!(diffusivities, closure::FlavorOfXKVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    Nx_in, Ny_in, Nz_in = total_size(diffusivities.κᶜ)
    ox_in, oy_in, oz_in = diffusivities.κᶜ.data.offsets

    kp = KernelParameters((Nx_in, Ny_in, Nz_in), (ox_in, oy_in, oz_in))

    launch!(arch, grid, kp,
            compute_ri_number!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy,
            top_tracer_bcs,
            clock)

    # Use `only_local_halos` to ensure that no communication occurs during
    # this call to fill_halo_regions!
    fill_halo_regions!(diffusivities.Ri; only_local_halos=true)

    launch!(arch, grid, kp,
            compute_xinkai_diffusivities!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy,
            top_tracer_bcs,
            clock)

    return nothing
end

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / S²

    # Clip N² and avoid NaN
    return ifelse(N² == 0, zero(grid), Ri)
end

const c = Center()
const f = Face()

@kernel function compute_ri_number!(diffusivities, grid, closure::FlavorOfXKVD,
                                    velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
end

@kernel function compute_xinkai_diffusivities!(diffusivities, grid, closure::FlavorOfXKVD,
                                                velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    _compute_xinkai_diffusivities!(i, j, k, diffusivities, grid, closure,
                                   velocities, tracers, buoyancy, tracer_bcs, clock)
end

@inline function _compute_xinkai_diffusivities!(i, j, k, diffusivities, grid, closure,
                                                velocities, tracers, buoyancy, tracer_bcs, clock)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀  = closure_ij.ν₀  
    νˢʰ = closure_ij.νˢʰ
    νᶜⁿ = closure_ij.νᶜⁿ
    Pr_convₜ = closure_ij.Pr_convₜ
    Pr_shearₜ = closure_ij.Pr_shearₜ
    Riᶜ = closure_ij.Riᶜ
    δRi = closure_ij.δRi

    κ₀ = ν₀ / Pr_shearₜ
    κˢʰ = νˢʰ / Pr_shearₜ
    κᶜⁿ = νᶜⁿ / Pr_convₜ

    # (Potentially) apply a horizontal filter to the Richardson number
    Ri = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)

    # Conditions
    convecting = Ri < 0 # applies regardless of Qᵇ

    # Convective adjustment diffusivity
    ν_local = ifelse(convecting, (νˢʰ - νᶜⁿ) * tanh(Ri / δRi) + νˢʰ, clamp((ν₀ - νˢʰ) * Ri / Riᶜ + νˢʰ, ν₀, νˢʰ))
    κ_local = ifelse(convecting, (κˢʰ - κᶜⁿ) * tanh(Ri / δRi) + κˢʰ, clamp((κ₀ - κˢʰ) * Ri / Riᶜ + κˢʰ, κ₀, κˢʰ))

    # Update by averaging in time
    @inbounds diffusivities.κᵘ[i, j, k] = ifelse(k <= 1 || k >= grid.Nz+1, 0, ν_local)
    @inbounds diffusivities.κᶜ[i, j, k] = ifelse(k <= 1 || k >= grid.Nz+1, 0, κ_local)

    return nothing
end
