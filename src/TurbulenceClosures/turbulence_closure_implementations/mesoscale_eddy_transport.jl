using Oceananigans.Fields: VelocityFields, ZeroField
using Oceananigans.Grids: inactive_node, peripheral_node
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b

struct MesoscaleEddyTransport{K, M, L, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, N}
                         κ :: K # Diffusivity at C, C, F locations
          isopycnal_tensor :: M
             slope_limiter :: L
    
    function MesoscaleEddyTransport{N}(κ :: K,
                                     isopycnal_tensor :: M,
                                     slope_limiter :: L) where {K, M, L, N}

        return new{K, M, L, N}(κ, isopycnal_tensor, slope_limiter)
    end
end

"""
    MesoscaleEddyTransport(FT = Float64; 
                           κ = 1000, 
                           isopycnal_tensor = SmallSlopeIsopycnalTensor(), 
                           slope_limiter = FluxTapering(100), 
                           required_halo_size::Int = 1)

Creates a `MesoscaleEddyTransport` turbulence closure that parameterized the transport of mesoscale eddies.
The eddy velocities are calculated from the buoyancy slope and the diffusivity as
```math
u = - ∂z (κ Sx)
v = - ∂z (κ Sy)
w = ∂x (κ Sx) + ∂y (κ Sy)
```
Where `κ` is provided (can be a `Function` or an `AbstractArray`) and 
```math
Sx = - ∂x b / ∂z b
Sy = - ∂y b / ∂z b
```
The eddy velocities are added to the model velocities to advect the tracer fields.
"""
function MesoscaleEddyTransport(FT = Float64;
                                κ = 1000,
                                isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                slope_limiter = FluxTapering(100),
                                required_halo_size::Int = 1) 

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    return MesoscaleEddyTransport{required_halo_size}(convert_diffusivity(FT, κ),
                                                    isopycnal_tensor,
                                                    slope_limiter)
end

Adapt.adapt_structure(to, closure::MesoscaleEddyTransport{N}) where N = 
    MesoscaleEddyTransport{N}(Adapt.adapt(to, closure.κ),
                              Adapt.adapt(to, closure.isopycnal_tensor),
                              Adapt.adapt(to, closure.slope_limiter))

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

# Slope in x-direction at F, C, F locations
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

# Slope in y-direction at F, C, F locations
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

@inline κ_Sxᶠᶜᶠ(i, j, k, grid, clk, clo, b, fields) = κᶠᶜᶠ(i, j, k, grid, CCF, clo.κ, clk.time, fields) * Sxᶠᶜᶠ(i, j, k, grid, clo, b, fields)
@inline κ_Syᶜᶠᶠ(i, j, k, grid, clk, clo, b, fields) = κᶜᶠᶠ(i, j, k, grid, CCF, clo.κ, clk.time, fields) * Syᶜᶠᶠ(i, j, k, grid, clo, b, fields)

@kernel function _compute_eddy_velocities!(uₑ, vₑ, wₑ, grid, clk, clo, b, fields)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        uₑ[i, j, k] = - ∂zᶠᶜᶜ(i, j, k, grid, κ_Sxᶠᶜᶠ, clk, clo, b, fields)
        vₑ[i, j, k] = - ∂zᶜᶠᶜ(i, j, k, grid, κ_Syᶜᶠᶠ, clk, clo, b, fields)

        wˣ = ∂xᶜᶜᶠ(i, j, k, grid, κ_Sxᶠᶜᶠ, clk, clo, b, fields)
        wʸ = ∂yᶜᶜᶠ(i, j, k, grid, κ_Syᶜᶠᶠ, clk, clo, b, fields) 
        
        wₑ[i, j, k] =  wˣ + wʸ
    end
end

# Single closure version
@inline closure_turbulent_velocity(clo, K, val_tracer_name) = (u = ZeroField(), v = ZeroField(), w = ZeroField())
@inline closure_turbulent_velocity(::MesoscaleEddyTransport, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

# 2-tuple closure
@inline select_velocities(::ZeroU, U) = U
@inline select_velocities(U, ::ZeroU) = U
@inline select_velocities(U::ZeroU, ::ZeroU) = U

# 3-tuple closure
@inline select_velocities(U, ::ZeroU, ::ZeroU) = U
@inline select_velocities(::ZeroU, U, ::ZeroU) = U
@inline select_velocities(::ZeroU, ::ZeroU, U) = U
@inline select_velocities(U::ZeroU, ::ZeroU, ::ZeroU) = U

# 4-tuple closure
@inline select_velocities(U, ::ZeroU, ::ZeroU, ::ZeroU) = U
@inline select_velocities(::ZeroU, U, ::ZeroU, ::ZeroU) = U
@inline select_velocities(::ZeroU, ::ZeroU, U, ::ZeroU) = U
@inline select_velocities(::ZeroU, ::ZeroU, ::ZeroU, U) = U
@inline select_velocities(U::ZeroU, ::ZeroU, ::ZeroU, ::ZeroU) = U

# Handle tuple of closures.
# Assumption: there is only one MesoscaleEddyTransport closure in the tuple.
@inline closure_turbulent_velocity(closures::Tuple{<:Any}, Ks, val_tracer_name) =
            closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name)

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any}, Ks, val_tracer_name) = 
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_turbulent_velocity(closures[4], Ks[4], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_turbulent_velocity(closures[4:end], Ks[4:end], val_tracer_name))