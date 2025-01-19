using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
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

struct XinKaiVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    ν₀  :: FT
    νˢʰ :: FT
    νᶜⁿ :: FT
    Cᵉⁿ :: FT
    Prₜ :: FT
    Riᶜ :: FT
    δRi :: FT
    Q₀  :: FT
    δQ  :: FT
end

function XinKaiVerticalDiffusivity{TD}(ν₀  :: FT, 
                                       νˢʰ :: FT,
                                       νᶜⁿ :: FT,
                                       Cᵉⁿ :: FT,
                                       Prₜ :: FT,
                                       Riᶜ :: FT,
				                       δRi :: FT,
                                       Q₀  :: FT,
	         		                   δQ  :: FT) where {TD, FT}
                                       
    return XinKaiVerticalDiffusivity{TD, FT}(ν₀, νˢʰ, νᶜⁿ, Cᵉⁿ, Prₜ, Riᶜ, δRi, Q₀, δQ)
end

function XinKaiVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                    FT  = Float64;
				                    ν₀  = 1e-5, 
                                    νˢʰ = 0.0885,
                                    νᶜⁿ = 4.3668,
                                    Cᵉⁿ = 0.2071,
                                    Prₜ = 1.207,
                                    Riᶜ = -0.21982,
				                    δRi = 8.342e-4,
                                    Q₀  = 0.08116,
	         		                δQ  = 0.02622) 

    TD = typeof(time_discretization)

    return XinKaiVerticalDiffusivity{TD}(convert(FT, ν₀),
                                         convert(FT, νˢʰ),
                                         convert(FT, νᶜⁿ),
                                         convert(FT, Cᵉⁿ),
                                         convert(FT, Prₜ),
                                         convert(FT, Riᶜ),
					                     convert(FT, δRi),
					                     convert(FT, Q₀),
					                     convert(FT, δQ))
end

XinKaiVerticalDiffusivity(FT::DataType; kw...) =
    XinKaiVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::XinKaiVerticalDiffusivity{TD, FT}) where {TD, FT} = 
    XinKaiVerticalDiffusivity{TD, FT}(clo.ν₀, clo.νˢʰ, clo.νᶜⁿ, clo.Cᵉⁿ, clo.Prₜ, clo.Riᶜ, clo.δRi, clo.Q₀, clo.δQ)   	
                                         
#####                                    
##### Diffusivity field utilities        
#####                                    
                                         
const RBVD = XinKaiVerticalDiffusivity   
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
    N² = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κᶜ, κᵘ, Ri, N²)
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

    launch!(arch, grid, kp, compute_N²!, diffusivities, grid, closure, tracers, buoyancy)
    launch!(arch, grid, kp, compute_ri_number!, diffusivities, grid, closure, velocities)

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

@inline function N²ᶜᶜᶠ(i, j, k, grid, buoyancy, tracers)
    return ∂z_b(i, j, k, grid, buoyancy, tracers)
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, diffusivities)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = diffusivities.N²[i, j, k]
    Ri = N² / S²

    # Clip N² and avoid NaN
    return ifelse(N² == 0, zero(grid), Ri)
end

const c = Center()
const f = Face()

@kernel function compute_N²!(diffusivities, grid, closure::FlavorOfXKVD, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.N²[i, j, k] = N²ᶜᶜᶠ(i, j, k, grid, buoyancy, tracers)
end

@kernel function compute_ri_number!(diffusivities, grid, closure::FlavorOfXKVD, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, diffusivities)
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
    Cᵉⁿ = closure_ij.Cᵉⁿ
    Prₜ = closure_ij.Prₜ
    Riᶜ = closure_ij.Riᶜ
    δRi = closure_ij.δRi
    Q₀  = closure_ij.Q₀ 
    δQ  = closure_ij.δQ 

    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))

    # (Potentially) apply a horizontal filter to the Richardson number
    Ri = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)
    Ri_above = ℑxyᶜᶜᵃ(i, j, k + 1, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)
    N² = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, diffusivities.N²)
    
    # Conditions
    convecting = Ri < 0 # applies regardless of Qᵇ
    entraining = (Ri > 0) & (Ri_above < 0) & (Qᵇ > 0)

    # Convective adjustment diffusivity
    ν_local = ifelse(convecting, - (νᶜⁿ - νˢʰ) / 2 * tanh(Ri / δRi) + νˢʰ, clamp(Riᶜ * Ri + νˢʰ + ν₀, ν₀, νˢʰ))

    # Entrainment diffusivity
    x = Qᵇ / (N² + 1e-11)
    ν_nonlocal = ifelse(entraining,  Cᵉⁿ * νᶜⁿ * 0.5 * (tanh((x - Q₀) / δQ) + 1), 0)

    # Update by averaging in time
    @inbounds diffusivities.κᵘ[i, j, k] = ifelse((k <= 1) | (k >= grid.Nz+1), 0, ν_local + ν_nonlocal)
    @inbounds diffusivities.κᶜ[i, j, k] = ifelse((k <= 1) | (k >= grid.Nz+1), 0, (ν_local + ν_nonlocal) / Prₜ)

    return nothing
end
