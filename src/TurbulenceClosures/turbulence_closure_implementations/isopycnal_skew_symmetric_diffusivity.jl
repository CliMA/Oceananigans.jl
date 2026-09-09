using Oceananigans.Fields: VelocityFields

struct AdvectiveFormulation end
struct DiffusiveFormulation end

struct IsopycnalSkewSymmetricDiffusivity{TD, A, K, S, M, L, N} <: AbstractTurbulenceClosure{TD, N}
    κ_skew :: K
    κ_symmetric :: S
    isopycnal_tensor :: M
    slope_limiter :: L

    function IsopycnalSkewSymmetricDiffusivity{TD, A, N}(κ_skew :: K,
                                                         κ_symmetric :: S,
                                                         isopycnal_tensor :: I,
                                                         slope_limiter :: L) where {TD, A, K, S, I, L, N}

        return new{TD, A, K, S, I, L, N}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter)
    end
end

const ISSD{TD, A} = IsopycnalSkewSymmetricDiffusivity{TD, A} where {TD, A}
const ISSDVector{TD, A} = AbstractVector{<:ISSD{TD, A}} where {TD, A}
const FlavorOfISSD{TD, A} = Union{ISSD{TD, A}, ISSDVector{TD, A}} where {TD, A}
const SkewAdvectionISSD = ISSD{<:Any, <:AdvectiveFormulation}

const issd_coefficient_loc = (Center(), Center(), Face())

# The slope limiter of Gerdes, Koberle and Willebrand (1991)
struct FluxTapering{FT}
    max_slope :: FT
end

"""
    IsopycnalSkewSymmetricDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                      κ_skew = 0,
                                      κ_symmetric = 0,
                                      skew_flux_formulation = DiffusiveFormulation(),
                                      isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                      slope_limiter = FluxTapering(1e-2),
                                      required_halo_size::Int = 1)

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.

Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.

The fluxes are discretized on the triads of Griffies et al. (1998), which makes the symmetric part exactly
adiabatic and the skew part exactly antisymmetric. With `skew_flux_formulation = AdvectiveFormulation()`
the skew part is instead carried by an eddy-induced velocity, which the tracer advection scheme transports.

This closure implements the mesoscale eddy parameterization developed by
[Gent and McWilliams (1990)](@cite GentMcWilliams90) and [Redi (1982)](@cite Redi82).

References
==========
* Griffies, S. M., A. Gnanadesikan, R. C. Pacanowski, V. D. Larichev, J. K. Dukowicz, and R. D. Smith (1998) Isoneutral diffusion in a z-coordinate ocean model. _J. Phys. Oceanogr._, **28**, 805–830, doi:10.1175/1520-0485(1998)028<0805:IDIAZC>2.0.CO;2
"""
function IsopycnalSkewSymmetricDiffusivity(time_disc::TD=VerticallyImplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType;
                                           κ_skew = 0,
                                           κ_symmetric = 0,
                                           skew_flux_formulation::A = DiffusiveFormulation(),
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = FluxTapering(1e-2),
                                           required_halo_size::Int = 1) where {TD, A}

    # For the moment, allow only one skew coefficient for all tracers
    if κ_skew isa NamedTuple && skew_flux_formulation isa AdvectiveFormulation
        error("Only one skew coefficient for all tracers is currently supported with the AdvectiveFormulation.")
    end

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    # `nothing` reads as "this part of the tensor is switched off"
    κ_skew = something(κ_skew, 0)
    κ_symmetric = something(κ_symmetric, 0)

    return IsopycnalSkewSymmetricDiffusivity{TD, A, required_halo_size}(convert_diffusivity(FT, κ_skew),
                                                                       convert_diffusivity(FT, κ_symmetric),
                                                                       isopycnal_tensor,
                                                                       slope_limiter)
end

IsopycnalSkewSymmetricDiffusivity(FT::DataType; kw...) =
    IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Utils.with_tracers(tracers, closure::ISSD{TD, A, <:Any, <:Any, <:Any, <:Any, N}) where {TD, A, N} =
    IsopycnalSkewSymmetricDiffusivity{TD, A, N}(closure.κ_skew, closure.κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)

# For ensembles of closures
function Utils.with_tracers(tracers, closure_vector::ISSDVector)
    arch = architecture(closure_vector)

    _closure_vector = arch isa Architectures.GPU ? Vector(closure_vector) : closure_vector

    Ex = length(_closure_vector)
    vec = [with_tracers(tracers, _closure_vector[i]) for i=1:Ex]

    return on_architecture(arch, vec)
end

function Adapt.adapt_structure(to, closure::ISSD{TD, A, <:Any, <:Any, <:Any, <:Any, N}) where {TD, A, N}
    return IsopycnalSkewSymmetricDiffusivity{TD, A, N}(Adapt.adapt(to, closure.κ_skew),
                                                       Adapt.adapt(to, closure.κ_symmetric),
                                                       Adapt.adapt(to, closure.isopycnal_tensor),
                                                       Adapt.adapt(to, closure.slope_limiter))
end

function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfISSD{TD, A}) where {TD, A}
    closure_fields = if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        (; ϵκR₃₃ = ZFaceField(grid))
    else
        NamedTuple()
    end

    if A() isa AdvectiveFormulation
        closure_fields = merge(closure_fields, VelocityFields(grid))
    end

    return closure_fields
end

function compute_closure_fields!(closure_fields, closure::FlavorOfISSD{TD}, model; parameters = :xyz) where TD

    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)

    if TD() isa VerticallyImplicitTimeDiscretization
        launch!(arch, grid, parameters,
                triad_compute_tapered_R₃₃!,
                closure_fields, grid, closure, clock, buoyancy, tracers)
    end

    compute_eddy_velocities!(closure_fields, closure, model; parameters)

    return nothing
end

@kernel function triad_compute_tapered_R₃₃!(K, grid, closure, clock, b, C)
    i, j, k, = @index(Global, NTuple)
    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    @inbounds K.ϵκR₃₃[i, j, k] = ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C)
end

#####
##### _triads_
#####
##### There are two horizontal slopes: Sx and Sy
#####
##### Both slopes are "located" at tracer cell centers.
#####
##### The slopes are computed by a directional derivative, which lends an
##### "orientation" to the slope. For example, the x-slope `Sx` computed
##### with a "+" directional derivative in x, and a "+" directional derivative
##### in z, is
#####
##### Sx⁺⁺ᵢₖ = Δz / Δx * (bᵢ₊₁ - bᵢ) / (bₖ₊₁ - bₖ)
#####
##### The superscript codes ⁺⁺, ⁺⁻, ⁻⁺, ⁻⁻, denote the direction of the derivative
##### in (h, z).
#####
##### from https://github.com/CliMA/Oceananigans.jl/blob/glw/homogeneous-bounded/src/TurbulenceClosures/turbulence_closure_implementations/isopycnal_potential_vorticity_diffusivity.jl
#####

@inline function triad_Sx(ix, iz, j, kx, kz, grid, buoyancy, tracers)
    bx = ∂xᵣ_b(ix, j, kx, grid, buoyancy, tracers)
    bz =  ∂z_b(iz, j, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(i, jy, jz, ky, kz, grid, buoyancy, tracers)
    by = ∂yᵣ_b(i, jy, ky, grid, buoyancy, tracers)
    bz =  ∂z_b(i, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - by / bz)
end

@inline Sx⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k+1, grid, buoyancy, tracers)
@inline Sx⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k,   grid, buoyancy, tracers)
@inline Sx⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   i, j, k, k+1, grid, buoyancy, tracers)
@inline Sx⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   i, j, k, k,   grid, buoyancy, tracers)

@inline Sy⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k+1, grid, buoyancy, tracers)
@inline Sy⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k,   grid, buoyancy, tracers)
@inline Sy⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   j, k, k+1, grid, buoyancy, tracers)
@inline Sy⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   j, k, k,   grid, buoyancy, tracers)

# We remove triads that live on a boundary (immersed or top / bottom / north / south / east / west)
@inline triad_mask_x(ix, iz, j, kx, kz, grid) = !peripheral_node(ix, j, kx, grid, Face(), Center(), Center()) & !peripheral_node(iz, j, kz, grid, Center(), Center(), Face())
@inline triad_mask_y(i, jy, jz, ky, kz, grid) = !peripheral_node(i, jy, ky, grid, Center(), Face(), Center()) & !peripheral_node(i, jz, kz, grid, Center(), Center(), Face())

# A triad standing on an unstratified vertical face carries no isoneutral flux.
@inline stably_stratified(i, j, k, grid, buoyancy, tracers) = ∂z_b(i, j, k, grid, buoyancy, tracers) > 0

# The limiter must bound the diffusivity each triad actually carries, `ϵ κ S²`, so `ϵ` is built from that triad's own slope.
@inline ϵx⁺⁺(i, j, k, grid, sl, b, C) = triad_mask_x(i+1, i, j, k, k+1, grid) * stably_stratified(i, j, k+1, grid, b, C) * tapering_factor(Sx⁺⁺(i, j, k, grid, b, C), zero(grid), sl)
@inline ϵx⁺⁻(i, j, k, grid, sl, b, C) = triad_mask_x(i+1, i, j, k, k,   grid) * stably_stratified(i, j, k,   grid, b, C) * tapering_factor(Sx⁺⁻(i, j, k, grid, b, C), zero(grid), sl)
@inline ϵx⁻⁺(i, j, k, grid, sl, b, C) = triad_mask_x(i,   i, j, k, k+1, grid) * stably_stratified(i, j, k+1, grid, b, C) * tapering_factor(Sx⁻⁺(i, j, k, grid, b, C), zero(grid), sl)
@inline ϵx⁻⁻(i, j, k, grid, sl, b, C) = triad_mask_x(i,   i, j, k, k,   grid) * stably_stratified(i, j, k,   grid, b, C) * tapering_factor(Sx⁻⁻(i, j, k, grid, b, C), zero(grid), sl)

@inline ϵy⁺⁺(i, j, k, grid, sl, b, C) = triad_mask_y(i, j+1, j, k, k+1, grid) * stably_stratified(i, j, k+1, grid, b, C) * tapering_factor(zero(grid), Sy⁺⁺(i, j, k, grid, b, C), sl)
@inline ϵy⁺⁻(i, j, k, grid, sl, b, C) = triad_mask_y(i, j+1, j, k, k,   grid) * stably_stratified(i, j, k,   grid, b, C) * tapering_factor(zero(grid), Sy⁺⁻(i, j, k, grid, b, C), sl)
@inline ϵy⁻⁺(i, j, k, grid, sl, b, C) = triad_mask_y(i, j,   j, k, k+1, grid) * stably_stratified(i, j, k+1, grid, b, C) * tapering_factor(zero(grid), Sy⁻⁺(i, j, k, grid, b, C), sl)
@inline ϵy⁻⁻(i, j, k, grid, sl, b, C) = triad_mask_y(i, j,   j, k, k,   grid) * stably_stratified(i, j, k,   grid, b, C) * tapering_factor(zero(grid), Sy⁻⁻(i, j, k, grid, b, C), sl)

@inline κˢ_κᴬᶜᶜᶜ(i, j, k, grid, loc, closure, clock, C) =
    (κᶜᶜᶜ(i, j, k, grid, loc, closure.κ_symmetric, clock, C),
     κᶜᶜᶜ(i, j, k, grid, loc, closure.κ_skew,      clock, C))

# The eddy-induced velocity already carries the skew flux
@inline κˢ_κᴬᶜᶜᶜ(i, j, k, grid, loc, closure::SkewAdvectionISSD, clock, C) =
    (κᶜᶜᶜ(i, j, k, grid, loc, closure.κ_symmetric, clock, C), zero(grid))

# Triad diagram key
# =================
#
#   * ┗ : Sx⁺⁺ / Sy⁺⁺
#   * ┛ : Sx⁻⁺ / Sy⁻⁺
#   * ┓ : Sx⁻⁻ / Sy⁻⁻
#   * ┏ : Sx⁺⁻ / Sy⁺⁻
#

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)
    sl = closure.slope_limiter
    loc = (Center(), Center(), Center())

    κˢ⁺, κᴬ⁺ = κˢ_κᴬᶜᶜᶜ(i-1, j, k, grid, loc, closure, clock, C)
    κˢ⁻, κᴬ⁻ = κˢ_κᴬᶜᶜᶜ(i,   j, k, grid, loc, closure, clock, C)

    ϵ⁺⁺ = ϵx⁺⁺(i-1, j, k, grid, sl, b, C)
    ϵ⁺⁻ = ϵx⁺⁻(i-1, j, k, grid, sl, b, C)
    ϵ⁻⁺ = ϵx⁻⁺(i,   j, k, grid, sl, b, C)
    ϵ⁻⁻ = ϵx⁻⁻(i,   j, k, grid, sl, b, C)

    # Small slope approximation
    ∂x_c = ∂xᵣᶠᶜᶜ(i, j, k, grid, c)

    #       i-1     i
    # k+1  -------------
    #           |      |
    #       ┏┗  ∘  ┛┓  | k
    #           |      |
    # k   ------|------|

    Fx = (ϵ⁺⁺ * (κˢ⁺ * ∂x_c + (κˢ⁺ - κᴬ⁺) * Sx⁺⁺(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k+1, grid, c)) +
          ϵ⁺⁻ * (κˢ⁺ * ∂x_c + (κˢ⁺ - κᴬ⁺) * Sx⁺⁻(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k,   grid, c)) +
          ϵ⁻⁺ * (κˢ⁻ * ∂x_c + (κˢ⁻ - κᴬ⁻) * Sx⁻⁺(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c)) +
          ϵ⁻⁻ * (κˢ⁻ * ∂x_c + (κˢ⁻ - κᴬ⁻) * Sx⁻⁻(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k,   grid, c))) / 4

    return - Fx
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)
    sl = closure.slope_limiter
    loc = (Center(), Center(), Center())

    κˢ⁺, κᴬ⁺ = κˢ_κᴬᶜᶜᶜ(i, j-1, k, grid, loc, closure, clock, C)
    κˢ⁻, κᴬ⁻ = κˢ_κᴬᶜᶜᶜ(i, j,   k, grid, loc, closure, clock, C)

    ϵ⁺⁺ = ϵy⁺⁺(i, j-1, k, grid, sl, b, C)
    ϵ⁺⁻ = ϵy⁺⁻(i, j-1, k, grid, sl, b, C)
    ϵ⁻⁺ = ϵy⁻⁺(i, j,   k, grid, sl, b, C)
    ϵ⁻⁻ = ϵy⁻⁻(i, j,   k, grid, sl, b, C)

    ∂y_c = ∂yᵣᶜᶠᶜ(i, j, k, grid, c)

    Fy = (ϵ⁺⁺ * (κˢ⁺ * ∂y_c + (κˢ⁺ - κᴬ⁺) * Sy⁺⁺(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k+1, grid, c)) +
          ϵ⁺⁻ * (κˢ⁺ * ∂y_c + (κˢ⁺ - κᴬ⁺) * Sy⁺⁻(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k,   grid, c)) +
          ϵ⁻⁺ * (κˢ⁻ * ∂y_c + (κˢ⁻ - κᴬ⁻) * Sy⁻⁺(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k+1, grid, c)) +
          ϵ⁻⁻ * (κˢ⁻ * ∂y_c + (κˢ⁻ - κᴬ⁻) * Sy⁻⁻(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k,   grid, c))) / 4

    return - Fy
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::FlavorOfISSD{TD}, K, ::Val{id},
                                  c, clock, C, b) where {TD, id}

    closure = getclosure(i, j, closure)
    sl = closure.slope_limiter
    loc = (Center(), Center(), Center())

    κˢ⁻, κᴬ⁻ = κˢ_κᴬᶜᶜᶜ(i, j, k,   grid, loc, closure, clock, C)
    κˢ⁺, κᴬ⁺ = κˢ_κᴬᶜᶜᶜ(i, j, k-1, grid, loc, closure, clock, C)

    ϵˣ⁻⁻ = ϵx⁻⁻(i, j, k,   grid, sl, b, C)
    ϵˣ⁺⁻ = ϵx⁺⁻(i, j, k,   grid, sl, b, C)
    ϵˣ⁻⁺ = ϵx⁻⁺(i, j, k-1, grid, sl, b, C)
    ϵˣ⁺⁺ = ϵx⁺⁺(i, j, k-1, grid, sl, b, C)

    ϵʸ⁻⁻ = ϵy⁻⁻(i, j, k,   grid, sl, b, C)
    ϵʸ⁺⁻ = ϵy⁺⁻(i, j, k,   grid, sl, b, C)
    ϵʸ⁻⁺ = ϵy⁻⁺(i, j, k-1, grid, sl, b, C)
    ϵʸ⁺⁺ = ϵy⁺⁺(i, j, k-1, grid, sl, b, C)

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

    κR₃₁_∂x_c = ((κˢ⁻ + κᴬ⁻) * ϵˣ⁻⁻ * Sx⁻⁻(i, j, k,   grid, b, C) * ∂xᵣᶠᶜᶜ(i,   j, k,   grid, c) +
                 (κˢ⁻ + κᴬ⁻) * ϵˣ⁺⁻ * Sx⁺⁻(i, j, k,   grid, b, C) * ∂xᵣᶠᶜᶜ(i+1, j, k,   grid, c) +
                 (κˢ⁺ + κᴬ⁺) * ϵˣ⁻⁺ * Sx⁻⁺(i, j, k-1, grid, b, C) * ∂xᵣᶠᶜᶜ(i,   j, k-1, grid, c) +
                 (κˢ⁺ + κᴬ⁺) * ϵˣ⁺⁺ * Sx⁺⁺(i, j, k-1, grid, b, C) * ∂xᵣᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    κR₃₂_∂y_c = ((κˢ⁻ + κᴬ⁻) * ϵʸ⁻⁻ * Sy⁻⁻(i, j, k,   grid, b, C) * ∂yᵣᶜᶠᶜ(i, j,   k,   grid, c) +
                 (κˢ⁻ + κᴬ⁻) * ϵʸ⁺⁻ * Sy⁺⁻(i, j, k,   grid, b, C) * ∂yᵣᶜᶠᶜ(i, j+1, k,   grid, c) +
                 (κˢ⁺ + κᴬ⁺) * ϵʸ⁻⁺ * Sy⁻⁺(i, j, k-1, grid, b, C) * ∂yᵣᶜᶠᶜ(i, j,   k-1, grid, c) +
                 (κˢ⁺ + κᴬ⁺) * ϵʸ⁺⁺ * Sy⁺⁺(i, j, k-1, grid, b, C) * ∂yᵣᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    κϵ_R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), clock, c, closure, b, C)

    return - κR₃₁_∂x_c - κR₃₂_∂y_c - κϵ_R₃₃_∂z_c
end

# The antisymmetric tensor has no 33 component, so only the symmetric diffusivity enters here.
@inline function ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C)
    loc = (Center(), Center(), Center())

    κ⁻ = κᶜᶜᶜ(i, j, k,   grid, loc, κ, clock, C)
    κ⁺ = κᶜᶜᶜ(i, j, k-1, grid, loc, κ, clock, C)

    ϵˣ⁻⁻ = ϵx⁻⁻(i, j, k,   grid, sl, b, C)
    ϵˣ⁺⁻ = ϵx⁺⁻(i, j, k,   grid, sl, b, C)
    ϵˣ⁻⁺ = ϵx⁻⁺(i, j, k-1, grid, sl, b, C)
    ϵˣ⁺⁺ = ϵx⁺⁺(i, j, k-1, grid, sl, b, C)

    ϵʸ⁻⁻ = ϵy⁻⁻(i, j, k,   grid, sl, b, C)
    ϵʸ⁺⁻ = ϵy⁺⁻(i, j, k,   grid, sl, b, C)
    ϵʸ⁻⁺ = ϵy⁻⁺(i, j, k-1, grid, sl, b, C)
    ϵʸ⁺⁺ = ϵy⁺⁺(i, j, k-1, grid, sl, b, C)

    ϵκR₃₃ = (κ⁻ * (ϵˣ⁻⁻ * Sx⁻⁻(i, j, k,   grid, b, C)^2 + ϵʸ⁻⁻ * Sy⁻⁻(i, j, k,   grid, b, C)^2  +
                   ϵˣ⁺⁻ * Sx⁺⁻(i, j, k,   grid, b, C)^2 + ϵʸ⁺⁻ * Sy⁺⁻(i, j, k,   grid, b, C)^2) +
             κ⁺ * (ϵˣ⁻⁺ * Sx⁻⁺(i, j, k-1, grid, b, C)^2 + ϵʸ⁻⁺ * Sy⁻⁺(i, j, k-1, grid, b, C)^2  +
                   ϵˣ⁺⁺ * Sx⁺⁺(i, j, k-1, grid, b, C)^2 + ϵʸ⁺⁺ * Sy⁺⁺(i, j, k-1, grid, b, C)^2)) / 4

    return ϵκR₃₃
end

@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, clock, c, closure, b, C)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    return ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C) * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, clock, c, closure, b, C) = zero(grid)

@inline κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id}, clock, fields) where id = @inbounds K.ϵκR₃₃[i, j, k]

@inline viscous_flux_ux(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_uy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_uz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

@inline viscous_flux_vx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_vy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

@inline viscous_flux_wx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

#####
##### Show
#####

Base.summary(closure::ISSD) = string("IsopycnalSkewSymmetricDiffusivity",
                                     "(κ_skew=",
                                     prettysummary(closure.κ_skew),
                                     ", κ_symmetric=", prettysummary(closure.κ_symmetric), ")")

function Base.show(io::IO, closure::ISSD{<:Any, A}) where A
    print(io, "IsopycnalSkewSymmetricDiffusivity:", '\n',
              "├── κ_skew: ", prettysummary(closure.κ_skew), '\n',
              "├── κ_symmetric: ", prettysummary(closure.κ_symmetric), '\n',
              "├── skew_flux_formulation: ", summary(A()), '\n',
              "├── isopycnal_tensor: ", summary(closure.isopycnal_tensor), '\n',
              "└── slope_limiter: ", summary(closure.slope_limiter))
end
