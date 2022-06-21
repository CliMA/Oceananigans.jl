#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

const two_32 = Int32(2)

const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

abstract type SmoothnessStencil end

struct VorticityStencil <:SmoothnessStencil end
struct VelocityStencil <:SmoothnessStencil end

struct WENO{N, FT, XT, YT, ZT, VI, WF, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N}
    
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

    "bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "advection scheme used near boundaries"
    boundary_scheme :: CA
    symmetric_scheme :: SI

    function WENO{N, FT, VI, WF}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                                 coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                                 coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                                 bounds::PP, boundary_scheme::CA,
                                 symmetric_scheme :: SI) where {N, FT, XT, YT, ZT, VI, WF, PP, CA, SI}

            return new{N, FT, XT, YT, ZT, VI, WF, PP, CA, SI}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                                              coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                                              coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                                              bounds, boundary_scheme, symmetric_scheme)
    end
end

WENO(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid = grid, kwargs...)

# Some usefull aliases
WENO3(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 3,  kwargs...)
WENO5(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 5,  kwargs...)
WENO7(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 7,  kwargs...)
WENO9(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 9,  kwargs...)
WENO11(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid = grid, order = 11, kwargs...)

function WENO(FT::DataType=Float64; 
               order = 5,
               grid = nothing, 
               zweno = true, 
               vector_invariant = nothing,
               bounds = nothing)
    
    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    if order < 3
        return UpwindBiasedFirstOrder()
    else
        VI = typeof(vector_invariant)
        N  = Int((order + 1) ÷ 2)

        weno_coefficients = compute_stretched_weno_coefficients(grid, false, FT; order = N)
        boundary_scheme = WENO(FT; grid, order = order - 2, zweno, vector_invariant, bounds)
        if N > 2
            symmetric_scheme = CenteredFourthOrder()
        else
            symmetric_scheme = CenteredSecondOrder()
        end
    end

    return WENO{N, FT, VI, zweno}(weno_coefficients[1:6]..., bounds, boundary_scheme, symmetric_scheme)
end

# Flavours of WENO
const ZWENO        = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}
const PositiveWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple}

const WENOVectorInvariantVel{N, FT, XT, YT, ZT, VI, WF, PP}  = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VelocityStencil, WF, PP}
const WENOVectorInvariantVort{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VorticityStencil, WF, PP}

const WENOVectorInvariant{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:SmoothnessStencil, WF, PP}

formulation(scheme::WENO) = scheme isa WENOVectorInvariant ? "Vector Invariant" : "Flux"

function Base.show(io::IO, a::WENO{N, FT, RX, RY, RZ}) where {N, FT, RX, RY, RZ}
    print(io, "WENO advection scheme order $(N*2 -1) and a $(formulation(a)) form: \n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, XT, YT, ZT, VI, WF, PP}) where {N, FT, XT, YT, ZT, VI, WF, PP} =
     WENO{N, FT, VI, WF}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                         Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                         Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                         Adapt.adapt(to, scheme.bounds),
                         Adapt.adapt(to, scheme.boundary_scheme),
                         Adapt.adapt(to, scheme.symmetric_scheme))

# pre-compute coefficients for stretched WENO
function compute_stretched_weno_coefficients(grid, stretched_smoothness, FT; order)
    
    rect_metrics = (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

    if grid isa Nothing
        @warn "defaulting to uniform WENO scheme with $(FT) precision, use WENO(grid = grid) if this was not intended"
        for metric in rect_metrics
            @eval $(Symbol(:coeff_ , metric)) = nothing
            @eval $(Symbol(:smooth_, metric)) = nothing
        end
    else
        !(grid isa RectilinearGrid) && (@warn "WENO on a curvilinear stretched coordinate is not validated, use at your own risk!!")

        metrics = return_metrics(grid)
        dirsize = (:Nx, :Nx, :Ny, :Ny, :Nz, :Nz)

        arch       = architecture(grid)
        Hx, Hy, Hz = halo_size(grid)
        new_grid   = with_halo((Hx+1, Hy+1, Hz+1), grid)

        for (dir, metric, rect_metric) in zip(dirsize, metrics, rect_metrics)
            @eval $(Symbol(:coeff_ , rect_metric)) = calc_interpolating_coefficients($FT, $new_grid.$metric, $arch, $new_grid.$dir; order = $order)
            @eval $(Symbol(:smooth_, rect_metric)) = calc_smoothness_coefficients($FT, $Val($stretched_smoothness), $new_grid.$metric, $arch, $new_grid.$dir; order = $order) 
        end
    end

    return (coeff_xᶠᵃᵃ , coeff_xᶜᵃᵃ , coeff_yᵃᶠᵃ , coeff_yᵃᶜᵃ , coeff_zᵃᵃᶠ , coeff_zᵃᵃᶜ ,
            smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ)
end

@inline calc_interpolating_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N; order) = nothing
@inline calc_interpolating_coefficients(FT, coord::AbstractRange, arch, N; order)                              = nothing
function calc_interpolating_coefficients(FT, coord, arch, N; order) 

    cpu_coord = arch_array(CPU(), coord)

    s = []
    for r in -1:order-1
        push!(s, create_interp_coefficients(FT, r, cpu_coord, arch, N; order))
    end

    return tuple(s...)
end

function create_interp_coefficients(FT, r, cpu_coord, arch, N; order)

    stencil = NTuple{order, FT}[]
    @inbounds begin
        for i = 0:N+1
            push!(stencil, stencil_coefficients(i, r, cpu_coord, cpu_coord; order))     
        end
    end
    return OffsetArray(arch_array(arch, stencil), -1)
end

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)
