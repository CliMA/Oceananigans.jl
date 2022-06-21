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
    "reconstruction scheme used for symmetric interpolation"
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

function WENO(FT::DataType=Float64; 
               order = 5,
               grid = nothing, 
               zweno = true, 
               vector_invariant = nothing,
               bounds = nothing)
    
    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if order < 3
        # WENO(order = 1) is equivalent to UpwindBiased(order = 1)
        return UpwindBiased(order = 1)
    else
        VI = typeof(vector_invariant)
        N  = Int((order + 1) ÷ 2)

        weno_coefficients = compute_reconstruction_coefficients(grid, FT, :WENO; order = N)
        boundary_scheme   = WENO(FT; grid, order = order - 2, zweno, vector_invariant, bounds)
        symmetric_scheme  = Centered(FT; grid, order = order - 1)
    end

    return WENO{N, FT, VI, zweno}(weno_coefficients..., bounds, boundary_scheme, symmetric_scheme)
end

WENO(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid = grid, kwargs...)

# Some usefull aliases
WENOThirdOrder(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 3,  kwargs...)
WENOFifthOrder(grid, FT::DataType=Float64;  kwargs...) = WENO(FT; grid = grid, order = 5,  kwargs...)

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

Base.summary(a::WENO{N}) where N = string("WENO reconstruction order ", N*2-1, " in ", formulation(a), " form")

Base.show(io::IO, a::WENO{N, FT, RX, RY, RZ}) where {N, FT, RX, RY, RZ} =
    print(io, summary(a), " \n",
              " Boundary scheme : ", "\n",
              "    └── ", summary(a.boundary_scheme) , "\n",
              " Symmetric scheme : ", "\n",
              "    └── ", summary(a.symmetric_scheme) , "\n",
              " Directions:", "\n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )

Adapt.adapt_structure(to, scheme::WENO{N, FT, XT, YT, ZT, VI, WF, PP}) where {N, FT, XT, YT, ZT, VI, WF, PP} =
     WENO{N, FT, VI, WF}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                         Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                         Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                         Adapt.adapt(to, scheme.bounds),
                         Adapt.adapt(to, scheme.boundary_scheme),
                         Adapt.adapt(to, scheme.symmetric_scheme))

# Retrieve precomputed coefficients (+2 for julia's 1 based indices)
@inline retrieve_coeff(scheme::WENO, r, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[r+2][i] 
