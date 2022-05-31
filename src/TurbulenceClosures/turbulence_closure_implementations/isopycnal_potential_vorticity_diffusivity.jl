using Oceananigans.Operators: ℑxzᶜᶜᶠ, ℑyzᶜᶜᶠ, ℑxyᶜᶠᶜ, ℑyzᶜᶠᶜ, ℑxyᶠᶜᶜ, ℑxzᶠᶜᶜ, ℑyzᶠᶠᶜ, ℑxzᶠᶠᶜ, ℑxyzᶠᶠᶜ, ℑyzᶠᶜᶠ, ℑxyᶜᶜᶜ, ℑyzᶜᶜᶜ, ℑxzᶜᶠᶠ, ℑxyzᶜᶠᶠ, ℑxyᶠᶜᶠ, ℑxyᶜᶠᶠ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Coriolis: fᶠᶠᵃ

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
    νz = nothing

    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the 33 component of the isopycnal rotation tensor
        R₃₃ = Field{Center, Center, Face}(grid)
        νz = Field{Center, Center, Face}(grid)
    end

    return (; R₃₃, νz)
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfIPVD{TD}, model) where TD

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy
    coriolis = model.coriolis

    if TD() isa VerticallyImplicitTimeDiscretization
        event = launch!(arch, grid, :xyz,
                        compute_νz_and_R₃₃!, diffusivities.R₃₃, diffusivities.νz, grid, closure, coriolis, buoyancy, tracers,
                        dependencies = device_event(arch))
    else
        event = NoneEvent()
    end

    wait(device(arch), event)

    return nothing
end

@kernel function compute_νz_and_R₃₃!(R₃₃, νz, grid, closure, coriolis, buoyancy, tracers)
    i, j, k, = @index(Global, NTuple)

    closure = getclosure(i, j, closure)

    Sx = Sxᶜᶜᶠ(i, j, k, grid, buoyancy, tracers)
    Sy = Syᶜᶜᶠ(i, j, k, grid, buoyancy, tracers)
    @inbounds R₃₃[i, j, k] = Sx^2 + Sy^2

    f = ℑxyᶜᶜᶠ(i, j, k, grid, fᶠᶠᵃ, coriolis)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    ϵ = one(grid) # tapering, Ry * bz / (Ry * by)
    @inbounds νz[i, j, k] = ifelse(N² == 0, zero(grid), ϵ * f^2 * closure.κ_potential_vorticity / N²)
end

# Triad diagram key
# =================
#
#   * ┗ : Sx⁺⁺ / Sy⁺⁺
#   * ┛ : Sx⁻⁺ / Sy⁻⁺
#   * ┓ : Sx⁻⁻ / Sy⁻⁻
#   * ┏ : Sx⁺⁻ / Sy⁺⁻
#

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid, closure::FlavorOfIPVD, K, ::Val{id}, U, C, clock, b) where id
    c = C[id]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, id)
    κ = κᶠᶜᶜ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    # Small slope approximation
    R₁₁_∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)
    R₁₂_∂y_c = zero(grid)

    #       i-1     i 
    # k+1  -------------
    #           |      |
    #       ┏┗  ∘  ┛┓  | k
    #           |      |
    # k   ------|------|

    R₁₃_∂z_c = (Sx⁺⁺(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k+1, grid, c) +
                Sx⁺⁻(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k,   grid, c) +
                Sx⁻⁺(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c) +
                Sx⁻⁻(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k,   grid, c)) / 4
    
    return - κ * (R₁₁_∂x_c + R₁₂_∂y_c + R₁₃_∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::FlavorOfIPVD, K, ::Val{id}, U, C, clock, b) where id

    c = C[id]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, id)
    κ = κᶜᶠᶜ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    # Small slope approximation
    R₂₁_∂x_c = zero(grid)
    R₂₂_∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    R₂₃_∂z_c = (Sy⁺⁺(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k+1, grid, c) +
                Sy⁺⁻(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k,   grid, c) +
                Sy⁻⁺(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k+1, grid, c) +
                Sy⁻⁻(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k,   grid, c)) / 4
    
    return - κ * (R₂₁_∂x_c + R₂₂_∂y_c + R₂₃_∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::FlavorOfIPVD{TD}, K, ::Val{id}, U, C, clock, b) where {TD, id}
    c = C[id]
    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_tracers, id)
    κ = κᶜᶜᶠ(i, j, k, grid, clock, ipvd_coefficient_loc, κ)

    # Triad diagram:
    #
    #   i-1    i    i+1
    # -------------------
    # |     |     |     |
    # |     | ┓ ┏ |  k  |
    # |     |     |     |
    # -  k  -- ∘ --     -
    # |     |     |     |
    # |     | ┛ ┗ | k-1 |
    # |     |     |     |
    # --------------------
    
    R₃₁_∂x_c = (Sx⁻⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i,   j, k,   grid, c) +
                Sx⁺⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k,   grid, c) +
                Sx⁻⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i,   j, k-1, grid, c) +
                Sx⁺⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    R₃₂_∂y_c = (Sy⁻⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j,   k,   grid, c) +
                Sy⁺⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k,   grid, c) +
                Sy⁻⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j,   k-1, grid, c) +
                Sy⁺⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), c, κ, closure, b, C)

    return - κ * (R₃₁_∂x_c + R₃₂_∂y_c + R₃₃_∂z_c)
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

# Defined at ccc
@inline function viscous_flux_ux(i, j, k, grid, closure::FlavorOfIPVD, K, U, C, clock, b)
    closure = getclosure(i, j, closure)
    κ = νᶜᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    # Small angle approximation
    R₁₁_∂x_u = ∂xᶜᶜᶜ(i, j, k, grid, U.u)
    R₁₂_∂y_u = zero(grid)

    #       ┏┗  ∘  ┛┓  | k
    #  ┓┏ 
    # ----
    #  ┛┗   
    #=
    R₁₃_∂z_u = (Sx⁺⁻(i, j, k, grid, b, C) * ∂zᶠᶜᶠ(i,   j, k,   grid, U.u) +
                Sx⁻⁻(i, j, k, grid, b, C) * ∂zᶠᶜᶠ(i-1, j, k,   grid, U.u) +
                Sx⁺⁺(i, j, k, grid, b, C) * ∂zᶠᶜᶠ(i,   j, k+1, grid, U.u) +
                Sx⁻⁺(i, j, k, grid, b, C) * ∂zᶠᶜᶠ(i-1, j, k+1, grid, U.u)) / 4
    =#
    R₁₃_∂z_u = zero(grid)

    return - κ * (R₁₁_∂x_u + R₁₂_∂y_u + R₁₃_∂z_u)
end

# Defined at ffc
@inline function viscous_flux_uy(i, j, k, grid, closure::FlavorOfIPVD, K, U, C, clock, b)
    closure = getclosure(i, j, closure)
    κ = νᶠᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    # Small angles
    R₂₁_∂x_u = zero(grid)
    R₂₂_∂y_u = ∂yᶠᶠᶜ(i, j, k, grid, U.u)

    #=
    R₂₃_∂z_u = (Sy⁻⁺(i,   j,   k, grid, b, C) * ∂zᶠᶜᶠ(i, j,   k+1, grid, U.u) +
                Sy⁻⁻(i,   j,   k, grid, b, C) * ∂zᶠᶜᶠ(i, j,   k,   grid, U.u) +
                Sy⁻⁺(i-1, j,   k, grid, b, C) * ∂zᶠᶜᶠ(i, j,   k+1, grid, U.u) +
                Sy⁻⁻(i-1, j,   k, grid, b, C) * ∂zᶠᶜᶠ(i, j,   k,   grid, U.u) +
                Sy⁺⁺(i,   j-1, k, grid, b, C) * ∂zᶠᶜᶠ(i, j-1, k+1, grid, U.u) +
                Sy⁺⁻(i,   j-1, k, grid, b, C) * ∂zᶠᶜᶠ(i, j-1, k,   grid, U.u) +
                Sy⁺⁺(i-1, j-1, k, grid, b, C) * ∂zᶠᶜᶠ(i, j-1, k+1, grid, U.u) +
                Sy⁺⁻(i-1, j-1, k, grid, b, C) * ∂zᶠᶜᶠ(i, j-1, k,   grid, U.u)) / 8
    =#
    R₂₃_∂z_u = zero(grid)

    return - κ * (R₂₁_∂x_u + R₂₂_∂y_u + R₂₃_∂z_u)
end

@inline function b_N⁻²_ccf(i, j, k, grid, buoyancy, tracers)
    b = ℑzᶜᶜᶠ(i, j, k, grid, buoyancy_perturbation, buoyancy, tracers)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(N² == 0, zero(grid), b / N²)
end

# Defined at fcf
@inline function viscous_flux_uz(i, j, k, grid, closure::FlavorOfIPVD{TD}, K, U, C, clock, b) where TD
    closure = getclosure(i, j, closure)
    κ = νᶠᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    #=
    R₃₁_∂x_u = (Sx⁻⁻(i,   j, k,   grid, b, C) * ∂xᶜᶜᶜ(i,   j, k,   grid, U.u) +
                Sx⁺⁻(i-1, j, k,   grid, b, C) * ∂xᶜᶜᶜ(i-1, j, k,   grid, U.u) +
                Sx⁻⁺(i,   j, k-1, grid, b, C) * ∂xᶜᶜᶜ(i,   j, k-1, grid, U.u) +
                Sx⁺⁺(i-1, j, k-1, grid, b, C) * ∂xᶜᶜᶜ(i-1, j, k-1, grid, U.u))

    R₃₂_∂y_u = (Sy⁻⁺(i,   j,   k, grid, b, C) * ∂yᶠᶠᶜ(i, j,   k+1, grid, U.u) +
                Sy⁻⁻(i,   j,   k, grid, b, C) * ∂yᶠᶠᶜ(i, j,   k,   grid, U.u) +
                Sy⁻⁺(i-1, j,   k, grid, b, C) * ∂yᶠᶠᶜ(i, j,   k+1, grid, U.u) +
                Sy⁻⁻(i-1, j,   k, grid, b, C) * ∂yᶠᶠᶜ(i, j,   k,   grid, U.u) +
                Sy⁺⁺(i,   j-1, k, grid, b, C) * ∂yᶠᶠᶜ(i, j-1, k+1, grid, U.u) +
                Sy⁺⁻(i,   j-1, k, grid, b, C) * ∂yᶠᶠᶜ(i, j-1, k,   grid, U.u) +
                Sy⁺⁺(i-1, j-1, k, grid, b, C) * ∂yᶠᶠᶜ(i, j-1, k+1, grid, U.u) +
                Sy⁺⁻(i-1, j-1, k, grid, b, C) * ∂yᶠᶠᶜ(i, j-1, k,   grid, U.u)) / 8
    =#
    R₃₁_∂x_u = zero(grid)
    R₃₂_∂y_u = zero(grid)
    R₃₃_∂z_u = explicit_R₃₃_∂z_u(i, j, k, grid, TD(), U.u, closure, K, b, C)

    # |---|---|---|
    # | ∘ | ∘ | ∘ |
    # |---|--fcf--|
    # | ∘ | ∘ | ∘ |
    # |---|---|---|
    
    #=
    Sy = (Sy⁺⁻(i,   j, k,   grid, b, C) +
          Sy⁻⁻(i,   j, k,   grid, b, C) +
          Sy⁺⁻(i-1, j, k,   grid, b, C) +
          Sy⁻⁻(i-1, j, k,   grid, b, C) +
          Sy⁺⁺(i,   j, k-1, grid, b, C) +
          Sy⁻⁺(i,   j, k-1, grid, b, C) +
          Sy⁺⁺(i-1, j, k-1, grid, b, C) +
          Sy⁻⁺(i-1, j, k-1, grid, b, C)) / 8   
    =#

    # by = ℑxyzᶠᶜᶠ(i, j, k, grid, ∂y_b, b, C)
    # bz =   ℑxᶠᶜᶠ(i, j, k, grid, ∂z_b, b, C)
    # Sy = ifelse(bz == 0, zero(grid), - by / bz)
    
    Sy = ℑxyᶠᶜᶠ(i, j, k, grid, ∂yᶜᶠᶠ, b_N⁻²_ccf, b, C)
    f_Sy = zero(grid) #closure.f * Sy

    return - κ * (R₃₁_∂x_u + R₃₂_∂y_u + R₃₃_∂z_u + 2 * f_Sy)
end

# Defined at ffc
@inline function viscous_flux_vx(i, j, k, grid, closure::FlavorOfIPVD, K, U, C, clock, b)
    closure = getclosure(i, j, closure)
    κ = νᶠᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    R₁₁_∂x_v = ∂xᶠᶠᶜ(i, j, k, grid, U.v)
    R₁₂_∂y_v = zero(grid)

    #R₁₃ = isopycnal_rotation_tensor_xz_ffc(i, j, k, grid, b, C, closure.isopycnal_tensor)
    #∂z_v = ℑxzᶠᶠᶜ(i, j, k, grid, ∂zᶜᶠᶠ, U.v)
    R₁₃_∂z_v = zero(grid)

    return - κ * (R₁₁_∂x_v + R₁₂_∂y_v + R₁₃_∂z_v)
end

# Defined at ccc
@inline function viscous_flux_vy(i, j, k, grid, closure::FlavorOfIPVD, K, U, C, clock, b)
    closure = getclosure(i, j, closure)
    κ = νᶜᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    R₁₁_∂x_v = zero(grid)
    R₁₂_∂y_v = ∂yᶜᶜᶜ(i, j, k, grid, U.v)
    R₁₃_∂z_v = zero(grid)

    return - κ * (R₁₁_∂x_v + R₁₂_∂y_v + R₁₃_∂z_v)
end

# Defined at cff
@inline function viscous_flux_vz(i, j, k, grid, closure::FlavorOfIPVD{TD}, K, U, C, clock, b) where TD
    closure = getclosure(i, j, closure)
    κ = νᶜᶠᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_potential_vorticity)

    R₃₁_∂x_v = zero(grid)
    R₃₂_∂y_v = zero(grid)
    R₃₃_∂z_v = explicit_R₃₃_∂z_v(i, j, k, grid, TD(), U.v, closure, K, b, C)

    #=
    Sx = (Sx⁺⁻(i, j,   k,   grid, b, C) +
          Sx⁻⁻(i, j,   k,   grid, b, C) +
          Sx⁺⁻(i, j-1, k,   grid, b, C) +
          Sx⁻⁻(i, j-1, k,   grid, b, C) +
          Sx⁺⁺(i, j,   k-1, grid, b, C) +
          Sx⁻⁺(i, j,   k-1, grid, b, C) +
          Sx⁺⁺(i, j-1, k-1, grid, b, C) +
          Sx⁻⁺(i, j-1, k-1, grid, b, C)) / 8   
    =#

    #bx = ℑxyzᶜᶠᶠ(i, j, k, grid, ∂x_b, b, C)
    #bz =   ℑyᶜᶠᶠ(i, j, k, grid, ∂z_b, b, C)
    #Sx = ifelse(bz == 0, zero(grid), - bx / bz)

    Sx = ℑxyᶜᶠᶠ(i, j, k, grid, ∂xᶠᶜᶠ, b_N⁻²_ccf, b, C)
    f_Sx = zero(grid) #closure.f * Sx

    return - κ * (R₃₁_∂x_v + R₃₂_∂y_v + R₃₃_∂z_v - 2 * f_Sx)
end

# ccf
@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, c, closure, K, b, C)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)
    Sx = Sxᶜᶜᶠ(i, j, k, grid, b, C)
    Sy = Syᶜᶜᶠ(i, j, k, grid, b, C)
    R₃₃ = Sx^2 + Sy^2
    return R₃₃ * ∂z_c
end

@inline function κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id}, clock) where id
    closure = getclosure(i, j, closure)
    κc = get_tracer_κ(closure.κ_tracers, id)
    R₃₃ = @inbounds K.R₃₃[i, j, k]
    return R₃₃ * κᶜᶜᶠ(i, j, k, grid, clock, ipvd_coefficient_loc, κc)
end

# fcf
@inline function explicit_R₃₃_∂z_u(i, j, k, grid, ::ExplicitTimeDiscretization, u, closure, K, b, C)
    R₃₃_∂z_u = zero(grid)
    return R₃₃_∂z_u
end

# cff
@inline function explicit_R₃₃_∂z_v(i, j, k, grid, ::ExplicitTimeDiscretization, v, closure, K, b, C)
    R₃₃_∂z_v = zero(grid)
    return R₃₃_∂z_v
end

# @inline νzᶜᶜᶜ(i, j, k, grid, clo::IPVD, K, clock) = ℑzᶜᶜᶜ(i, j, k, grid, K.νz)
# @inline νzᶠᶠᶜ(i, j, k, grid, clo::IPVD, K, clock) = ℑxyzᶠᶠᶜ(i, j, k, grid, K.νz)
@inline νzᶠᶜᶠ(i, j, k, grid, clo::IPVD, K, clock) = ℑxᶠᶜᶠ(i, j, k, grid, K.νz)
@inline νzᶜᶠᶠ(i, j, k, grid, clo::IPVD, K, clock) = ℑyᶜᶠᶠ(i, j, k, grid, K.νz)

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)
@inline explicit_R₃₃_∂z_u(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)
@inline explicit_R₃₃_∂z_v(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)

@inline viscous_flux_wx(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::FlavorOfIPVD, args...) = zero(grid)

