#####
##### Weighted Essentially Non-Oscillatory (WENO) fifth-order advection scheme
#####

using OffsetArrays
using Oceananigans.Grids: with_halo
using Oceananigans.Architectures: arch_array, architecture
using Adapt
import Base: show

const two_32 = Int32(2)

const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

abstract type SmoothnessStencil end

struct VorticityStencil <:SmoothnessStencil end
struct VelocityStencil <:SmoothnessStencil end

"""
    struct WENO5{FT, XT, YT, ZT, XS, YS, ZS, WF} <: AbstractUpwindBiasedAdvectionScheme{2}

Weighted Essentially Non-Oscillatory (WENO) fifth-order advection scheme.

$(TYPEDFIELDS)
"""
struct WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, WF} <: AbstractUpwindBiasedAdvectionScheme{2}
    "coefficient for ENO reconstruction on x-faces" 
    coeff_xᶠᵃᵃ::XT
    "coefficient for ENO reconstruction on x-centers"
    coeff_xᶜᵃᵃ::XT
    "coefficient for ENO reconstruction on y-faces"
    coeff_yᵃᶠᵃ::YT
    "coefficient for ENO reconstruction on y-centers"
    coeff_yᵃᶜᵃ::YT
    "coefficient for ENO reconstruction on z-faces"
    coeff_zᵃᵃᶠ::ZT
    "coefficient for ENO reconstruction on z-centers"
    coeff_zᵃᵃᶜ::ZT
    
    "coefficient for WENO smoothness indicators on x-faces"
    smooth_xᶠᵃᵃ::XS
    "coefficient for WENO smoothness indicators on x-centers"
    smooth_xᶜᵃᵃ::XS
    "coefficient for WENO smoothness indicators on y-faces"
    smooth_yᵃᶠᵃ::YS
    "coefficient for WENO smoothness indicators on y-centers"
    smooth_yᵃᶜᵃ::YS
    "coefficient for WENO smoothness indicators on z-faces"
    smooth_zᵃᵃᶠ::ZS
    "coefficient for WENO smoothness indicators on z-centers"
    smooth_zᵃᵃᶜ::ZS

    "coefficient for WENO reconstruction, optimal weights"
    C3₀ :: FT
    C3₁ :: FT 
    C3₂ :: FT
end

"""
    WENO5([FT = Float64;] grid = nothing, stretched_smoothness = false, zweno = false, vector_invariant = nothing)

Construct a fifth-order weigthed essentially non-oscillatory advection scheme. The constructor allows
construction of WENO schemes on either uniform or stretched grids.

Keyword arguments
=================

  - `grid`: (defaults to `nothing`)
  - `stretched_smoothness`: When `true` it results in computing the coefficients for the smoothness
    indicators β₀, β₁ and β₂ so that they account for the stretched `grid`. (defaults to `false`)
  - `zweno`: When `true` implement a Z-WENO formulation for the WENO weights calculation. (defaults to
    `false`)
  - `vector_invariant`:

Not providing any keyword argument, `WENO5()` defaults to the uniform 5th-order coefficients ("uniform
setting) in all directions, using a Z-WENO formulation.

```jldoctest; filter = [r".*┌ Warning.*", r".*└ @ Oceananigans.*"]
julia> using Oceananigans

julia> WENO5()
┌ Warning: defaulting to uniform WENO scheme with Float64 precision, use WENO5(grid = grid) if this was not intended
└ @ Oceananigans.Advection .../src/Advection/weno_fifth_order.jl:90
WENO5 advection scheme with:
    ├── X regular
    ├── Y regular
    └── Z regular
```

`WENO5(grid = grid)` defaults to uniform interpolation coefficient for each of the grid directions that
is uniform (`typeof(Δc) <: Number`) while it precomputes the ENO coefficients for reconstruction for all
grid directions that are stretched. (After testing "on-the-fly" calculation of coefficients for stretched
directions ended up being way too expensive and, therefore, is not supported.)

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (3, 4, 5), x = (0, 1), y = (0, 1), z = [-10, -9, -7, -4, -1.5, 0]);

julia> WENO5(grid = grid)
WENO5 advection scheme with:
    ├── X regular
    ├── Y regular
    └── Z stretched
```

`WENO5(grid = grid, stretched_smoothness = true)` behaves similarly to `WENO5(grid = grid)` but,
additionally, it also computes the smoothness indicators coefficients, ``β₀``, ``β₁``, and ``β₂``,
taking into account the stretched dimensions.

`WENO5(zweno = false)` implements a JS-WENO formulation for the WENO weights calculation

Comments
========

All methods have the roughly the same execution speed except for `stretched_smoothness = true` that
requires more memory and is less computationally efficient, especially on GPUs. In addition, it has
not been found to be much impactful on the tested cases. As such, most of the times we urge users
to use `WENO5(grid = grid)`, as this increases accuracy on a stretched mesh  but does decreases
memory utilization (and also results in a slight speed-up).

(The above claims were made after some preliminary tests. Thus, we still users to perform some
benchmarks/checks before performing, e.g., a large simulation on a "weirdly" stretched grid.)

On the other hand, a Z-WENO formulation is *most of the times* beneficial (also in case of a uniform
mesh) with roughly the same performances (just a slight slowdown). The same can be said for the
stretched `WENO5(grid = grid)` formulation in case of stretched grids.

References
==========

Shu, Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic
    Conservation Laws, 1997, NASA/CR-97-206253, ICASE Report No. 97-65

Castro et al, High order weighted essentially non-oscillatory WENO-Z schemes for hyperbolic conservation
    laws, 2011, Journal of Computational Physics, 230(5), 1766-1792
"""
function WENO5(coeffs = nothing, FT = Float64; 
               grid = nothing, 
               stretched_smoothness = false, 
               zweno = true, 
               vector_invariant = nothing)
    
    rect_metrics = (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

    if grid isa Nothing
        @warn "defaulting to uniform WENO scheme with $(FT) precision, use WENO5(grid = grid) if this was not intended"
        for metric in rect_metrics
            @eval $(Symbol(:coeff_ , metric)) = nothing
            @eval $(Symbol(:smooth_, metric)) = nothing
        end
    else
        !(grid isa RectilinearGrid) && (@warn "WENO on a curvilinear stretched coordinate is not validated, use at your own risk!!")

        metrics      = return_metrics(grid)
        dirsize      = (:Nx, :Nx, :Ny, :Ny, :Nz, :Nz)

        FT       = eltype(grid)
        arch     = architecture(grid)
        new_grid = with_halo((4, 4, 4), grid)
       
        for (dir, metric, rect_metric) in zip(dirsize, metrics, rect_metrics)
            @eval $(Symbol(:coeff_ , rect_metric)) = calc_interpolating_coefficients($FT, $new_grid.$metric, $arch, $new_grid.$dir)
            @eval $(Symbol(:smooth_, rect_metric)) = calc_smoothness_coefficients($FT, $Val($stretched_smoothness), $new_grid.$metric, $arch, $new_grid.$dir) 
        end
    end

    XT = typeof(coeff_xᶠᵃᵃ)
    YT = typeof(coeff_yᵃᶠᵃ)
    ZT = typeof(coeff_zᵃᵃᶠ)
    XS = typeof(smooth_xᶠᵃᵃ)
    YS = typeof(smooth_yᵃᶠᵃ)
    ZS = typeof(smooth_zᵃᵃᶠ)

    if coeffs isa Nothing
        C3₀, C3₁, C3₂ = FT.((3/10, 3/5, 1/10))
    else
        C3₀, C3₁, C3₂ = FT.(coeffs)
    end

    VI = typeof(vector_invariant)

    return WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, zweno}(coeff_xᶠᵃᵃ , coeff_xᶜᵃᵃ , coeff_yᵃᶠᵃ , coeff_yᵃᶜᵃ , coeff_zᵃᵃᶠ , coeff_zᵃᵃᶜ ,
                                                        smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ,
                                                        C3₀, C3₁, C3₂)
end

return_metrics(::LatitudeLongitudeGrid) = (:λᶠᵃᵃ, :λᶜᵃᵃ, :φᵃᶠᵃ, :φᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)
return_metrics(::RectilinearGrid)       = (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

# Flavours of WENO
const ZWENO = WENO5{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}

const WENOVectorInvariantVel{FT, XT, YT, ZT, XS, YS, ZS, VI, WF}  = 
      WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, WF} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:VelocityStencil, WF}

const WENOVectorInvariantVort{FT, XT, YT, ZT, XS, YS, ZS, VI, WF} = 
      WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, WF} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:VorticityStencil, WF}

const WENOVectorInvariant = WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, WF} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:SmoothnessStencil, WF}

function Base.show(io::IO, a::WENO5{FT, RX, RY, RZ}) where {FT, RX, RY, RZ}
    print(io, "WENO5 advection scheme with: \n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )
end

Adapt.adapt_structure(to, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS, VI, WF}) where {FT, XT, YT, ZT, XS, YS, ZS, VI, WF} =
     WENO5{FT, typeof(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ)),
               typeof(Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ)),  
               typeof(Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ)),
               typeof(Adapt.adapt(to, scheme.smooth_xᶠᵃᵃ)),
               typeof(Adapt.adapt(to, scheme.smooth_yᵃᶠᵃ)),  
               typeof(Adapt.adapt(to, scheme.smooth_zᵃᵃᶠ)), VI, WF}(
        Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ),
        Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
        Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ),
        Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
        Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ),       
        Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
        Adapt.adapt(to, scheme.smooth_xᶠᵃᵃ),
        Adapt.adapt(to, scheme.smooth_xᶜᵃᵃ),
        Adapt.adapt(to, scheme.smooth_yᵃᶠᵃ),
        Adapt.adapt(to, scheme.smooth_yᵃᶜᵃ),
        Adapt.adapt(to, scheme.smooth_zᵃᵃᶠ),       
        Adapt.adapt(to, scheme.smooth_zᵃᵃᶜ), scheme.C3₀, scheme.C3₁, scheme.C3₂)

@inline boundary_buffer(::WENO5) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

# Stencil to calculate the stretched WENO weights and smoothness indicators
@inline left_stencil_x(i, j, k, ψ, args...) = @inbounds ( (ψ[i-3, j, k], ψ[i-2, j, k], ψ[i-1, j, k]), (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]) )
@inline left_stencil_y(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-3, k], ψ[i, j-2, k], ψ[i, j-1, k]), (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]) )
@inline left_stencil_z(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-3], ψ[i, j, k-2], ψ[i, j, k-1]), (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]) )

@inline right_stencil_x(i, j, k, ψ, args...) = @inbounds ( (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]), (ψ[i, j, k], ψ[i+1, j, k], ψ[i+2, j, k]) )
@inline right_stencil_y(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]), (ψ[i, j, k], ψ[i, j+1, k], ψ[i, j+2, k]) )
@inline right_stencil_z(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]), (ψ[i, j, k], ψ[i, j, k+1], ψ[i, j, k+2]) )

@inline left_stencil_x(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i-3, j, k, args...), ψ(i-2, j, k, args...), ψ(i-1, j, k, args...)), (ψ(i-2, j, k, args...), ψ(i-1, j, k, args...), ψ(i, j, k, args...)), (ψ(i-1, j, k, args...), ψ(i, j, k, args...), ψ(i+1, j, k, args...)) )
@inline left_stencil_y(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j-3, k, args...), ψ(i, j-2, k, args...), ψ(i, j-1, k, args...)), (ψ(i, j-2, k, args...), ψ(i, j-1, k, args...), ψ(i, j, k, args...)), (ψ(i, j-1, k, args...), ψ(i, j, k, args...), ψ(i, j+1, k, args...)) )
@inline left_stencil_z(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j, k-3, args...), ψ(i, j, k-2, args...), ψ(i, j, k-1, args...)), (ψ(i, j, k-2, args...), ψ(i, j, k-1, args...), ψ(i, j, k, args...)), (ψ(i, j, k-1, args...), ψ(i, j, k, args...), ψ(i, j, k+1, args...)) )

@inline right_stencil_x(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i-2, j, k, args...), ψ(i-1, j, k, args...), ψ(i, j, k, args...)), (ψ(i-1, j, k, args...), ψ(i, j, k, args...), ψ(i+1, j, k, args...)), (ψ(i, j, k, args...), ψ(i+1, j, k, args...), ψ(i+2, j, k, args...)) )
@inline right_stencil_y(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j-2, k, args...), ψ(i, j-1, k, args...), ψ(i, j, k, args...)), (ψ(i, j-1, k, args...), ψ(i, j, k, args...), ψ(i, j+1, k, args...)), (ψ(i, j, k, args...), ψ(i, j+1, k, args...), ψ(i, j+2, k, args...)) )
@inline right_stencil_z(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j, k-2, args...), ψ(i, j, k-1, args...), ψ(i, j, k, args...)), (ψ(i, j, k-1, args...), ψ(i, j, k, args...), ψ(i, j, k+1, args...)), (ψ(i, j, k, args...), ψ(i, j, k+1, args...), ψ(i, j, k+2, args...)) )

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction

# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_left_stencil_u(i, j, k, ::Val{1}, u)  = @inbounds left_stencil_x(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_u(i, j, k, ::Val{2}, u)  = @inbounds left_stencil_y(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_v(i, j, k, ::Val{1}, v)  = @inbounds left_stencil_x(i, j, k, ℑxᶠᵃᵃ, v)
@inline tangential_left_stencil_v(i, j, k, ::Val{2}, v)  = @inbounds left_stencil_y(i, j, k, ℑxᶠᵃᵃ, v)

@inline tangential_right_stencil_u(i, j, k, ::Val{1}, u)  = @inbounds right_stencil_x(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_u(i, j, k, ::Val{2}, u)  = @inbounds right_stencil_y(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_v(i, j, k, ::Val{1}, v)  = @inbounds right_stencil_x(i, j, k, ℑxᶠᵃᵃ, v)
@inline tangential_right_stencil_v(i, j, k, ::Val{2}, v)  = @inbounds right_stencil_y(i, j, k, ℑxᶠᵃᵃ, v)

#####
##### biased pₖ for û calculation
#####

@inline left_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_left_p₀(scheme, args...) .* ψ)
@inline left_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_left_p₁(scheme, args...) .* ψ)
@inline left_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_left_p₂(scheme, args...) .* ψ)

@inline right_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_right_p₀(scheme, args...) .* ψ)
@inline right_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_right_p₁(scheme, args...) .* ψ)
@inline right_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_right_p₂(scheme, args...) .* ψ)

#####
##### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998)
#####

@inline left_biased_β₀(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, ψ, ::Type{Nothing}, scheme, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

#####
##### Stretched smoothness indicators gathered from precomputed values.
##### The stretched values for β coefficients are calculated from 
##### Shu, NASA/CR-97-206253, ICASE Report No. 97-65
##### by hardcoding that p(x) is a 2nd order polynomial
#####

@inline function biased_left_β(ψ, scheme, r, dir, i, location) 
    @inbounds begin
        stencil = retrieve_left_smooth(scheme, r, dir, i, location)
        wᵢᵢ = stencil[1]   
        wᵢⱼ = stencil[2]
        # horrible but have to do this for GPU execution (broadcast doesn't work apparently)
        result = 0
        for j = 1:3
            result += ψ[j] * ( wᵢᵢ[j] * ψ[j] + wᵢⱼ[j] * dagger(ψ)[j] )
        end
    end
    return result
end

@inline function biased_right_β(ψ, scheme, r, dir, i, location) 
    @inbounds begin
        stencil = retrieve_right_smooth(scheme, r, dir, i, location)
        wᵢᵢ = stencil[1]   
        wᵢⱼ = stencil[2]
        # horrible but have to do this for GPU execution (broadcast doesn't work apparently sum(ψ.*(wᵢᵢ.*ψ.+wᵢⱼ.*dagger(ψ))) )
        result = 0
        for j = 1:3
            result += ψ[j] * ( wᵢᵢ[j] * ψ[j] + wᵢⱼ[j] * dagger(ψ)[j] )
        end
    end
    return result
end

@inline left_biased_β₀(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 0, args...) 
@inline left_biased_β₁(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 1, args...) 
@inline left_biased_β₂(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 2, args...) 

@inline right_biased_β₀(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 2, args...) 
@inline right_biased_β₁(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 1, args...) 
@inline right_biased_β₂(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 0, args...) 

#####
##### VectorInvariant reconstruction (based on JS or Z) (z-direction Val{3} is different from x- and y-directions)
#####
##### Z-WENO-5 reconstruction (Castro et al: High order weighted essentially non-oscillatory WENO-Z schemes for hyperbolic conservation laws)
#####
##### JS-WENO-5 reconstruction
#####

for (side, coeffs) in zip([:left, :right], ([:C3₀, :C3₁, :C3₂], [:C3₂, :C3₁, :C3₀]))
    biased_weno5_weights = Symbol(side, :_biased_weno5_weights)
    biased_β₀ = Symbol(side, :_biased_β₀)
    biased_β₁ = Symbol(side, :_biased_β₁)
    biased_β₂ = Symbol(side, :_biased_β₂)
    
    tangential_stencil_u = Symbol(:tangential_, side, :_stencil_u)
    tangential_stencil_v = Symbol(:tangential_, side, :_stencil_v)

    biased_stencil_z = Symbol(side, :_stencil_z)
    
    @eval begin
        @inline function $biased_weno5_weights(FT, ψₜ, T, scheme, dir, idx, loc, args...)
            ψ₂, ψ₁, ψ₀ = ψₜ 
            β₀ = $biased_β₀(FT, ψ₀, T, scheme, dir, idx, loc)
            β₁ = $biased_β₁(FT, ψ₁, T, scheme, dir, idx, loc)
            β₂ = $biased_β₂(FT, ψ₂, T, scheme, dir, idx, loc)
            
            if scheme isa ZWENO
                τ₅ = abs(β₂ - β₀)
                α₀ = scheme.$(coeffs[1]) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
                α₁ = scheme.$(coeffs[2]) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
                α₂ = scheme.$(coeffs[3]) * (1 + (τ₅ / (β₂ + FT(ε)))^ƞ) 
            else
                α₀ = scheme.$(coeffs[1]) / (β₀ + FT(ε))^ƞ
                α₁ = scheme.$(coeffs[2]) / (β₁ + FT(ε))^ƞ
                α₂ = scheme.$(coeffs[3]) / (β₂ + FT(ε))^ƞ
            end
        
            Σα = α₀ + α₁ + α₂
            w₀ = α₀ / Σα
            w₁ = α₁ / Σα
            w₂ = α₂ / Σα
        
            return w₀, w₁, w₂
        end

        @inline function $biased_weno5_weights(FT, ijk, T, scheme, dir, idx, loc, ::Type{VelocityStencil}, u, v)
            i, j, k = ijk
            
            u₂, u₁, u₀ = $tangential_stencil_u(i, j, k, dir, u)
            v₂, v₁, v₀ = $tangential_stencil_v(i, j, k, dir, v)
        
            βu₀ = $biased_β₀(FT, u₀, T, scheme, Val(2), idx, loc)
            βu₁ = $biased_β₁(FT, u₁, T, scheme, Val(2), idx, loc)
            βu₂ = $biased_β₂(FT, u₂, T, scheme, Val(2), idx, loc)
        
            βv₀ = $biased_β₀(FT, v₀, T, scheme, Val(1), idx, loc)
            βv₁ = $biased_β₁(FT, v₁, T, scheme, Val(1), idx, loc)
            βv₂ = $biased_β₂(FT, v₂, T, scheme, Val(1), idx, loc)
                   
            β₀ = 0.5*(βu₀ + βv₀)  
            β₁ = 0.5*(βu₁ + βv₁)     
            β₂ = 0.5*(βu₂ + βv₂)  
        
            if scheme isa ZWENO
                τ₅ = abs(β₂ - β₀)
                α₀ = scheme.$(coeffs[1]) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
                α₁ = scheme.$(coeffs[2]) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
                α₂ = scheme.$(coeffs[3]) * (1 + (τ₅ / (β₂ + FT(ε)))^ƞ) 
            else    
                α₀ = scheme.$(coeffs[1]) / (β₀ + FT(ε))^ƞ
                α₁ = scheme.$(coeffs[2]) / (β₁ + FT(ε))^ƞ
                α₂ = scheme.$(coeffs[3]) / (β₂ + FT(ε))^ƞ
            end
                
            Σα = α₀ + α₁ + α₂
            w₀ = α₀ / Σα
            w₁ = α₁ / Σα
            w₂ = α₂ / Σα
        
            return w₀, w₁, w₂
        end

        @inline function $biased_weno5_weights(FT, ijk, T, scheme, ::Val{3}, idx, loc, ::Type{VelocityStencil}, u)
            i, j, k = ijk
            
            u₂, u₁, u₀ = $biased_stencil_z(i, j, k, u)
        
            β₀ = $biased_β₀(FT, u₀, T, scheme, Val(3), idx, loc)
            β₁ = $biased_β₁(FT, u₁, T, scheme, Val(3), idx, loc)
            β₂ = $biased_β₂(FT, u₂, T, scheme, Val(3), idx, loc)
        
            if scheme isa ZWENO
                τ₅ = abs(β₂ - β₀)
                α₀ = scheme.$(coeffs[1]) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
                α₁ = scheme.$(coeffs[2]) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
                α₂ = scheme.$(coeffs[3]) * (1 + (τ₅ / (β₂ + FT(ε)))^ƞ) 
            else    
                α₀ = scheme.$(coeffs[1]) / (β₀ + FT(ε))^ƞ
                α₁ = scheme.$(coeffs[2]) / (β₁ + FT(ε))^ƞ
                α₂ = scheme.$(coeffs[3]) / (β₂ + FT(ε))^ƞ
            end
                
            Σα = α₀ + α₁ + α₂
            w₀ = α₀ / Σα
            w₁ = α₁ / Σα
            w₂ = α₂ / Σα
        
            return w₀, w₁, w₂
        end
    end
end

#####
##### Biased interpolation functions
#####

pass_stencil(ψ, i, j, k, stencil) = ψ 
pass_stencil(ψ, i, j, k, ::Type{VelocityStencil}) = (i, j, k)

for (interp, dir, val, cT, cS) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT], [:XS, :YS, :ZS]) 
    for side in (:left, :right)
        interpolate_func = Symbol(:weno_, side, :_biased_interpolate_, interp)
        stencil       = Symbol(side, :_stencil_, dir)
        weno5_weights = Symbol(side, :_biased_weno5_weights)
        biased_p₀ = Symbol(side, :_biased_p₀)
        biased_p₁ = Symbol(side, :_biased_p₁)
        biased_p₂ = Symbol(side, :_biased_p₂)

        @eval begin
            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, 
                                               ψ, idx, loc, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
                
                ψ₂, ψ₁, ψ₀ = ψₜ = $stencil(i, j, k, ψ, grid, args...)
                w₀, w₁, w₂ = $weno5_weights(FT, pass_stencil(ψₜ, i, j, k, Nothing), $cS, scheme, Val($val), idx, loc, Nothing, args...)
                return w₀ * $biased_p₀(scheme, ψ₀, $cT, Val($val), idx, loc) + 
                       w₁ * $biased_p₁(scheme, ψ₁, $cT, Val($val), idx, loc) + 
                       w₂ * $biased_p₂(scheme, ψ₂, $cT, Val($val), idx, loc)
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENOVectorInvariant{FT, XT, YT, ZT, XS, YS, ZS}, 
                                               ψ, idx, loc, VI, args...) where {FT, XT, YT, ZT, XS, YS, ZS}

                ψ₂, ψ₁, ψ₀ = ψₜ = $stencil(i, j, k, ψ, grid, args...)
                w₀, w₁, w₂ = $weno5_weights(FT, pass_stencil(ψₜ, i, j, k, VI), $cS, scheme, Val($val), idx, loc, VI, args...)
                return w₀ * $biased_p₀(scheme, ψ₀, $cT, Val($val), idx, loc) + 
                       w₁ * $biased_p₁(scheme, ψ₁, $cT, Val($val), idx, loc) + 
                       w₂ * $biased_p₂(scheme, ψ₂, $cT, Val($val), idx, loc)
            end
        end
    end
end

#####
##### Coefficients for stretched (and uniform) ENO schemes (see Shu NASA/CR-97-206253, ICASE Report No. 97-65)
#####

@inline coeff_left_p₀(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),    FT(5/6), - FT(1/6))
@inline coeff_left_p₁(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (- FT(1/6),    FT(5/6),   FT(1/3))
@inline coeff_left_p₂(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),  - FT(7/6),  FT(11/6))

@inline coeff_right_p₀(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₂(scheme, Nothing, args...)) 
@inline coeff_right_p₁(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₁(scheme, Nothing, args...)) 
@inline coeff_right_p₂(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₀(scheme, Nothing, args...)) 

@inline coeff_left_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 0, dir, i ,loc)
@inline coeff_left_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 1, dir, i ,loc)
@inline coeff_left_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 2, dir, i ,loc)

@inline coeff_right_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, -1, dir, i ,loc)
@inline coeff_right_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  0, dir, i ,loc)
@inline coeff_right_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  1, dir, i ,loc)

@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.coeff_xᶠᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.coeff_xᶜᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.coeff_yᵃᶠᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.coeff_yᵃᶜᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.coeff_zᵃᵃᶠ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.coeff_zᵃᵃᶜ[r+2][i] 

@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+1][i] 

@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+4][i] 

@inline calc_interpolating_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_interpolating_coefficients(FT, coord::AbstractRange, arch, N)                              = nothing

@inline calc_smoothness_coefficients(FT, ::Val{false}, args...) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::AbstractRange, arch, N) = nothing


function calc_interpolating_coefficients(FT, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    s1 = create_interp_coefficients(FT,-1, cpu_coord, arch, N)
    s2 = create_interp_coefficients(FT, 0, cpu_coord, arch, N)
    s3 = create_interp_coefficients(FT, 1, cpu_coord, arch, N)
    s4 = create_interp_coefficients(FT, 2, cpu_coord, arch, N)

    return (s1, s2, s3, s4)
end

function create_interp_coefficients(FT, r, cpu_coord, arch, N)

    stencil = NTuple{3, FT}[]
    @inbounds begin
        for i = 0:N+1
            push!(stencil, interp_weights(r, cpu_coord, i, 0, -))     
        end
    end
    return OffsetArray(arch_array(arch, stencil), -1)
end

function calc_smoothness_coefficients(FT, beta, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    s1 = create_smoothness_coefficients(FT, 0, -, cpu_coord, arch, N)
    s2 = create_smoothness_coefficients(FT, 1, -, cpu_coord, arch, N)
    s3 = create_smoothness_coefficients(FT, 2, -, cpu_coord, arch, N)
    s4 = create_smoothness_coefficients(FT, 0, +, cpu_coord, arch, N)
    s5 = create_smoothness_coefficients(FT, 1, +, cpu_coord, arch, N)
    s6 = create_smoothness_coefficients(FT, 2, +, cpu_coord, arch, N)
    
    return (s1, s2, s3, s4, s5, s6)
end

function create_smoothness_coefficients(FT, r, op, cpu_coord, arch, N)

    # derivation written on overleaf
    
    stencil = NTuple{2, NTuple{3, FT}}[]   
    @inbounds begin
        for i = 0:N+1
       
            bias1 = Int(op == +)
            bias2 = bias1 - 1

            Δcᵢ = cpu_coord[i + bias1] - cpu_coord[i + bias2]
        
            Bᵢ  = prim_interp_weights(r, cpu_coord, i, bias1, op)
            bᵢ  =      interp_weights(r, cpu_coord, i, bias1, op)
            bₓᵢ = der1_interp_weights(r, cpu_coord, i, bias1, op)
            Aᵢ  = prim_interp_weights(r, cpu_coord, i, bias2, op)
            aᵢ  =      interp_weights(r, cpu_coord, i, bias2, op)
            aₓᵢ = der1_interp_weights(r, cpu_coord, i, bias2, op)

            pₓₓ = der2_interp_weights(r, cpu_coord, i, op)
            Pᵢ  =  (Bᵢ .- Aᵢ)

            wᵢᵢ = Δcᵢ  .* (bᵢ .* bₓᵢ .- aᵢ .* aₓᵢ .- pₓₓ .* Pᵢ)  .+ Δcᵢ^4 .* (pₓₓ .* pₓₓ)
            wᵢⱼ = Δcᵢ  .* (star(bᵢ, bₓᵢ)  .- star(aᵢ, aₓᵢ) .- star(pₓₓ, Pᵢ)) .+
                                                 Δcᵢ^4 .* star(pₓₓ, pₓₓ)

            push!(stencil, (wᵢᵢ, wᵢⱼ))
        end
    end

    return OffsetArray(arch_array(arch, stencil), -1)
end

@inline dagger(ψ)    = (ψ[2], ψ[3], ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

# Integral of ENO coefficients for 2nd order polynomial reconstruction at the face
function prim_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        sum  = 0 
                        for q = 0:3
                            if q != m && q != l 
                                prod *= coord[op(i, r-q+1)]
                                sum  += coord[op(i, r-q+1)]
                            end
                        end
                        num += coord[i+bias]^3 / 3 - sum * coord[i+bias]^2 / 2 + prod * coord[i+bias]
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# Second derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der2_interp_weights(r, coord, i, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        num += 2 
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# first derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der1_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        sum = 0
                        for q = 0:3
                            if q != m && q != l 
                                sum += coord[op(i, r-q+1)]
                            end
                        end
                        num += 2 * coord[i+bias] - sum
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# ENO coefficients for 2nd order polynomial reconstruction at the face
function interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        for q = 0:3
                            if q != m && q != l 
                                prod *= (coord[i+bias] - coord[op(i, r-q+1)])
                            end
                        end
                        num += prod
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end



