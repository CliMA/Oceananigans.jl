using Oceananigans.Fields: VelocityFields
using Oceananigans.Grids: inactive_node, peripheral_node
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b

struct MesoscaleEddyTransport{K, M, L, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, N}
                         κ :: K
          isopycnal_tensor :: M
             slope_limiter :: L
    
    function MesoscaleEddyTransport{N}(κ :: K,
                                     isopycnal_tensor :: M,
                                     slope_limiter :: L) where {K, M, L, N}

        return new{K, M, L, N}(κ, isopycnal_tensor, slope_limiter)
    end
end

function MesoscaleEddyTransport(FT = Float64;
                                κ = 1000,
                                isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                slope_limiter = FluxTapering(100),
                                required_halo_size::Int = 1) 

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    return MesoscaleEddyTransport{required_halo_size}(convert_diffusivity(FT, κ_skew),
                                                    isopycnal_tensor,
                                                    slope_limiter)
end

DiffusivityFields(grid, tracer_names, bcs, closure::MesoscaleEddyTransport) = VelocityFields(grid)
with_tracers(tracer_names, closure::MesoscaleEddyTransport) = closure

function compute_diffusivities!(diffusivities, closure::MesoscaleEddyTransport, model; parameters = :xyz)
    uₑ = diffusivities.u
    vₑ = diffusivities.v
    wₑ = diffusivities.w

    buoyancy = model.buoyancy
    grid     = model.grid
    clock    = model.clock

    model_fields = fields(model)
    
    launch!(architecture(grid), grid, parameters, _compute_eddy_velocities!, 
            uₑ, vₑ, wₑ, grid, clock, closure, buoyancy, model_fields)
    
    return nothing
end

@inline function Sxᶠᶜᶠ(i, j, k, grid, clo, b, C)
    bx = ℑzᵃᵃᶠ(i, j, k, grid, ∂x_b, b, C) 
    bz = ℑxᶠᵃᵃ(i, j, k, grid, ∂z_b, b, C)

    Sₘ = clo.slope_limiter.max_slope

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Face(), Center(), Face())
    Sx = ifelse(bz == 0,  zero(grid), - bx / bz)
    ϵ  = min(one(grid), Sₘ^2 / Sx^2)
    Sx = ifelse(inactive, zero(grid), Sx)

    return ϵ * Sx
end

@inline function Syᶜᶠᶠ(i, j, k, grid, clo, b, C)
    by = ℑzᵃᵃᶠ(i, j, k, grid, ∂y_b, b, C) 
    bz = ℑyᵃᶠᵃ(i, j, k, grid, ∂z_b, b, C)
    
    Sₘ = clo.slope_limiter.max_slope

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Center(), Face(), Face())
    Sy = ifelse(bz == 0,  zero(grid), - by / bz)
    ϵ  = min(one(grid), Sₘ^2 / Sy^2)
    Sy = ifelse(inactive, zero(grid), Sy)

    return ϵ * Sy
end 

const CCF = (Center, Center, Face)

@inline κSxᶠᶜᶠ(i, j, k, grid, clk, clo, b, fields) = κᶠᶜᶠ(i, j, k, grid, CCF, clo.κ, clk.time, fields) * Sxᶠᶜᶠ(i, j, k, grid, clo, b, fields)
@inline κSyᶜᶠᶠ(i, j, k, grid, clk, clo, b, fields) = κᶜᶠᶠ(i, j, k, grid, CCF, clo.κ, clk.time, fields) * Syᶜᶠᶠ(i, j, k, grid, clo, b, fields)

@kernel function _compute_eddy_velocities!(uₑ, vₑ, wₑ, grid, clk, clo, b, fields)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        uₑ[i, j, k] = - ∂zᶠᶜᶜ(i, j, k, grid, κSxᶠᶜᶠ, clk, clo, b, fields)
        vₑ[i, j, k] = - ∂zᶜᶠᶜ(i, j, k, grid, κSyᶜᶠᶠ, clk, clo, b, fields)

        wˣ = ∂xᶜᶜᶠ(i, j, k, grid, κSxᶠᶜᶠ, clk, clo, b, fields)
        wʸ = ∂yᶜᶜᶠ(i, j, k, grid, κSyᶜᶠᶠ, clk, clo, b, fields) 
        
        wₑ[i, j, k] =  wˣ + wʸ
    end
end

# Fallback
@inline closure_turbulent_velocity(clo, K, val_tracer_name) = (u = ZeroField(), v = ZeroField(), w = ZeroField())
@inline closure_turbulent_velocity(::MesoscaleEddyTransport, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

# Handle tuple of closures.
@inline function closure_turbulent_velocity(clo::Tuple, K::Tuple, val_tracer_name) 

end
