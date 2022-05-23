using Oceananigans.Operators: ℑxzᶜᶜᶠ, ℑyzᶜᶜᶠ, ℑxyᶜᶠᶜ, ℑyzᶜᶠᶜ, ℑxyᶠᶜᶜ, ℑxzᶠᶜᶜ, ℑyzᶠᶠᶜ, ℑxzᶠᶠᶜ, ℑxyzᶠᶠᶜ, ℑyzᶠᶜᶠ, ℑxyᶜᶜᶜ, ℑyzᶜᶜᶜ, ℑxzᶜᶠᶠ, ℑxyzᶜᶠᶠ, ℑxyᶠᶜᶠ, ℑxyᶜᶠᶠ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

struct IsopycnalPotentialVorticityDiffusivity{TD, Q, K, M, F} <: AbstractTurbulenceClosure{TD}
    κ_potential_vorticity :: Q
    κ_tracers :: K
    isopycnal_tensor :: M
    f :: F
    
    function IsopycnalPotentialVorticityDiffusivity{TD}(κ_potential_vorticity :: Q,
                                                        κ_tracers :: K,
                                                        isopycnal_tensor :: M,
                                                        f :: F) where {TD, Q, K, M, F}

        return new{TD, Q, K, M, F}(κ_potential_vorticity, κ_tracers, isopycnal_tensor, f)
    end
end

const IPVD{TD, A} = IsopycnalPotentialVorticityDiffusivity{TD, A} where {TD, A}
const IPVDVector{TD, A} = AbstractVector{<:IPVD{TD, A}} where {TD, A}
const FlavorOfIPVD{TD, A} = Union{IPVD{TD, A}, IPVDVector{TD, A}} where {TD, A}
const ipvd_coefficient_loc = (Center(), Center(), Center())

"""
    IsopycnalPotentialVorticityDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                           κ_potential_vorticity = 0,
                                           κ_tracers = 0,
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = nothing)
"""
function IsopycnalPotentialVorticityDiffusivity(time_disc::TD = VerticallyImplicitTimeDiscretization(), FT = Float64;
                                                κ_potential_vorticity = 0,
                                                κ_tracers = 0,
                                                f = 0.0,
                                                isopycnal_tensor = SmallSlopeIsopycnalTensor()) where {TD}

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    return IsopycnalPotentialVorticityDiffusivity{TD}(κ_potential_vorticity,
                                                      κ_tracers,
                                                      isopycnal_tensor,
                                                      f)
end

IsopycnalPotentialVorticityDiffusivity(FT::DataType; kw...) = 
    IsopycnalPotentialVorticityDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

function with_tracers(tracers, closure::IPVD{TD}) where TD
    κ_tracers = !isa(closure.κ_tracers, NamedTuple) ? closure.κ_tracers : tracer_diffusivities(tracers, closure.κ_tracers)
    return IsopycnalPotentialVorticityDiffusivity{TD}(closure.κ_potential_vorticity,
                                                      κ_tracers,
                                                      closure.isopycnal_tensor,
                                                      closure.f)
end

# For ensembles of closures
function with_tracers(tracers, closure_vector::IPVDVector)
    arch = architecture(closure_vector)

    if arch isa Architectures.GPU
        closure_vector = Vector(closure_vector)
    end

    Ex = length(closure_vector)
    closure_vector = [with_tracers(tracers, closure_vector[i]) for i=1:Ex]

    return arch_array(arch, closure_vector)
end

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfIPVD{TD}) where TD

    R₃₃ = nothing

    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the 33 component of the isopycnal rotation tensor
        R₃₃ = Field{Center, Center, Face}(grid)
    end

    return (; R₃₃)
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfIPVD{TD}, model) where TD

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    if TD() isa VerticallyImplicitTimeDiscretization
        R³³_event = launch!(arch, grid, :xyz,
                            compute_R₃₃!, diffusivities.R₃₃, grid, closure, tracers, buoyancy,
                            dependencies = device_event(arch))
    else
        R³³_event = NoneEvent()
    end

    wait(device(arch), R³³_event)

    return nothing
end

@kernel function compute_R₃₃!(R₃₃, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    @inbounds R₃₃[i, j, k] = 0.0 #isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure_ij.isopycnal_tensor)
end

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{IPVD, IPVDVector}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where tracer_index

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, tracer_index)
    κ = κᶠᶜᶜ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂y_c = ℑxyᶠᶜᶜ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ℑxzᶠᶜᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    
    return - κ * (R₁₁ * ∂x_c + R₁₂ * ∂y_c + R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::Union{IPVD, IPVDVector}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where tracer_index

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, tracer_index)
    κ = κᶜᶠᶜ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂x_c = ℑxyᶜᶠᶜ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂z_c = ℑyzᶜᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)
    
    R₂₁ = zero(grid)
    R₂₂ = one(grid)
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    return - κ * (R₂₁ * ∂x_c + R₂₂ * ∂y_c + R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::FlavorOfIPVD{TD}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where {tracer_index, TD}

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, tracer_index)
    κ = κᶜᶜᶠ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    # Average... of... the gradient!
    ∂x_c = ℑxzᶜᶜᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᶜᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    κ_∂z_c = explicit_κ_∂z_c(i, j, k, grid, TD(), c, κ, closure, buoyancy, tracers)

    return - κ_∂z_c - κ * (R₃₁ * ∂x_c + R₃₂ * ∂y_c)
end

@inline function κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfIPVD, K, ::Val{id}, clock) where id
    closure = getclosure(i, j, closure)
    κ = get_tracer_κ(closure.κ_tracers, id)
    R₃₃ = @inbounds K.R₃₃[i, j, k] # 33 component of rotation tensor
    return R₃₃ * κᶜᶜᶠ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)
end

#####
##### Redi diffusion for momentum.
#####

function b_N²_ccf(i, j, k, grid, buoyancy, tracers)
    b = ℑzᶜᶜᶠ(i, j, k, grid, buoyancy_perturbation, buoyancy, tracers)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return b * N²
end

# Defined at ccc
@inline function viscous_flux_ux(i, j, k, grid, closure::FlavorOfIPVD, K, U, tracers, clock, buoyancy)
    closure = getclosure(i, j, closure)
    κ = νᶜᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_u = ∂xᶜᶜᶜ(i, j, k, grid, U.u)
    ∂y_u = ℑxyᶜᶜᶜ(i, j, k, grid, ∂yᶠᶠᶜ, U.u)
    ∂z_u = ℑxzᶜᶜᶜ(i, j, k, grid, ∂zᶠᶜᶠ, U.u)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = zero(grid) #isopycnal_rotation_tensor_xz_ccc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    return - κ * (R₁₁ * ∂x_u + R₁₂ * ∂y_u + R₁₃ * ∂z_u)
end

# Defined at ffc
@inline function viscous_flux_uy(i, j, k, grid, closure::FlavorOfIPVD, K, U, tracers, clock, buoyancy)
    closure = getclosure(i, j, closure)
    κ = νᶠᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_u = ℑxyᶠᶠᶜ(i, j, k, grid, ∂xᶜᶜᶜ, U.u)
    ∂y_u = ∂yᶠᶠᶜ(i, j, k, grid, U.u)
    ∂z_u = ℑyzᶠᶠᶜ(i, j, k, grid, ∂zᶠᶜᶠ, U.u)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = zero(grid) #isopycnal_rotation_tensor_yz_ffc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    return - κ * (R₁₁ * ∂x_u + R₁₂ * ∂y_u + R₁₃ * ∂z_u)
end

# Defined at fcf
@inline function viscous_flux_uz(i, j, k, grid, closure::FlavorOfIPVD{TD}, K, U, tracers, clock, buoyancy) where TD
    closure = getclosure(i, j, closure)
    κ = νᶠᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_u = ℑxzᶠᶜᶠ(i, j, k, grid, ∂xᶜᶜᶜ, U.u)
    ∂y_u = ℑyzᶠᶜᶠ(i, j, k, grid, ∂yᶠᶠᶜ, U.u)

    R₃₁ = zero(grid) #isopycnal_rotation_tensor_xz_fcf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₂ = zero(grid) #isopycnal_rotation_tensor_yz_fcf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    κ_∂z_u = explicit_κ_∂z_u(i, j, k, grid, TD(), U.u, κ, closure, buoyancy, tracers)

    Sy = ℑxyᶠᶜᶠ(i, j, k, grid, ∂yᶜᶠᶠ, b_N²_ccf, buoyancy, tracers)
    f_Sy = closure.f * Sy

    #return - κ * f_Sy
    #return - κ_∂z_u - κ * (R₃₁ * ∂x_u + R₃₂ * ∂y_u)
    return - κ_∂z_u - κ * (R₃₁ * ∂x_u + R₃₂ * ∂y_u + 2 * f_Sy)
end

# Defined at ffc
@inline function viscous_flux_vx(i, j, k, grid, closure::FlavorOfIPVD, K, U, tracers, clock, buoyancy)
    closure = getclosure(i, j, closure)
    κ = νᶠᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_v = ∂xᶠᶠᶜ(i, j, k, grid, U.v)
    ∂y_v = ℑxyᶠᶠᶜ(i, j, k, grid, ∂yᶜᶜᶜ, U.v)
    ∂z_v = ℑxzᶠᶠᶜ(i, j, k, grid, ∂zᶜᶠᶠ, U.v)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = zero(grid) #isopycnal_rotation_tensor_xz_ffc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    return - κ * (R₁₁ * ∂x_v + R₁₂ * ∂y_v + R₁₃ * ∂z_v)
end

# Defined at ccc
@inline function viscous_flux_vy(i, j, k, grid, closure::FlavorOfIPVD, K, U, tracers, clock, buoyancy)
    closure = getclosure(i, j, closure)
    κ = νᶜᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_v = ℑxyᶜᶜᶜ(i, j, k, grid, ∂xᶠᶠᶜ, U.v)
    ∂y_v = ∂yᶜᶜᶜ(i, j, k, grid, U.v)
    ∂z_v = ℑyzᶜᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, U.v)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = zero(grid) #isopycnal_rotation_tensor_yz_ffc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    return - κ * (R₁₁ * ∂x_v + R₁₂ * ∂y_v + R₁₃ * ∂z_v)
end

# Defined at cff
@inline function viscous_flux_vz(i, j, k, grid, closure::FlavorOfIPVD{TD}, K, U, tracers, clock, buoyancy) where TD
    closure = getclosure(i, j, closure)
    κ = νᶜᶠᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)
    ∂x_v = ℑxzᶜᶠᶠ(i, j, k, grid, ∂xᶠᶠᶜ, U.v)
    ∂y_v = ℑyzᶜᶠᶠ(i, j, k, grid, ∂yᶜᶜᶜ, U.v)

    R₃₁ = zero(grid) #isopycnal_rotation_tensor_xz_cff(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₂ = zero(grid) #isopycnal_rotation_tensor_yz_cff(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    κ_∂z_v = explicit_κ_∂z_v(i, j, k, grid, TD(), U.v, κ, closure, buoyancy, tracers)

    Sx = ℑxyᶜᶠᶠ(i, j, k, grid, ∂xᶠᶜᶠ, b_N²_ccf, buoyancy, tracers)
    f_Sx = closure.f * Sx

    #return κ * f_Sx
    #return - κ_∂z_v - κ * (R₃₁ * ∂x_v + R₃₂ * ∂y_v)
    return - κ_∂z_v - κ * (R₃₁ * ∂x_v + R₃₂ * ∂y_v - 2 * f_Sx)
end

@inline function explicit_κ_∂z_u(i, j, k, grid, ::ExplicitTimeDiscretization, u, κ, closure, buoyancy, tracers)
    ∂z_u = ∂zᶠᶜᶠ(i, j, k, grid, u)
    R₃₃ = isopycnal_rotation_tensor_zz_fcf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    return κ * R₃₃ * ∂z_u
end

@inline function explicit_κ_∂z_v(i, j, k, grid, ::ExplicitTimeDiscretization, v, κ, closure, buoyancy, tracers)
    ∂z_v = ∂zᶜᶠᶠ(i, j, k, grid, v)
    R₃₃ = isopycnal_rotation_tensor_zz_cff(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    return κ * R₃₃ * ∂z_v
end

@inline explicit_κ_∂z_u(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)
@inline explicit_κ_∂z_v(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)

@inline viscous_flux_wx(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)

