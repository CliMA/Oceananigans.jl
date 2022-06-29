#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

abstract type SmoothnessStencil end

struct VorticityStencil <:SmoothnessStencil end
struct VelocityStencil <:SmoothnessStencil end

struct WENO{N, FT, XT, YT, ZT, VI, WF, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    
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

"""
    WENO(FT::DataType=Float64; 
         order = 5,
         grid = nothing, 
         zweno = true, 
         vector_invariant = nothing,
         bounds = nothing)
               
Construct a weigthed essentially non-oscillatory advection scheme of order `order`.

Keyword arguments
=================

- `order`: The order of the WENO advection scheme.
- `grid`: (defaults to `nothing`)
- `vector_invariant`: The stencil for which the vector-invariant form of the advection
                      scheme would use. Options `VelocityStencil()` or `VorticityStencil()`;
                      defaults to `nothing`.

- `zweno`: When `true` implement a Z-WENO formulation for the WENO weights calculation.
           (defaults to `false`)

Examples
========
```jldoctest
julia> WENO()
WENO reconstruction order 5 in Flux form
 Boundary scheme:
    └── WENO reconstruction order 3 in Flux form
 Symmetric scheme:
    └── Centered reconstruction order 4
 Directions:
    ├── X regular
    ├── Y regular
    └── Z regular
```

```jldoctest
julia> Nx, Nz = 16, 10;

julia> Lx, Lz = 1e4, 1e3;

julia> chebychev_spaced_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

julia> grid = RectilinearGrid(size = (Nx, Nz), topology=(Periodic, Flat, Bounded),
                              x = (0, Lx), z = chebychev_spaced_z_faces);

julia> WENO(order=5; grid)
WENO reconstruction order 5 in Flux form
 Smoothness formulation:
    └── Z-weno
 Boundary scheme:
    └── WENO reconstruction order 3 in Flux form
 Symmetric scheme:
    └── Centered reconstruction order 4
 Directions:
    ├── X regular
    ├── Y regular
    └── Z stretched
```
"""
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

WENO(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid, kwargs...)

# Some usefull aliases
WENOThirdOrder(grid=nothing, FT::DataType=Float64;  kwargs...) = WENO(grid, FT; order=3, kwargs...)
WENOFifthOrder(grid=nothing, FT::DataType=Float64;  kwargs...) = WENO(grid, FT; order=5, kwargs...)

# Flavours of WENO
const ZWENO        = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}
const PositiveWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple}

const WENOVectorInvariantVel{N, FT, XT, YT, ZT, VI, WF, PP}  = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VelocityStencil, WF, PP}
const WENOVectorInvariantVort{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VorticityStencil, WF, PP}

const WENOVectorInvariant{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:SmoothnessStencil, WF, PP}

formulation(scheme::WENO)                = "Flux form"
formulation(scheme::WENOVectorInvariant) = "Vector Invariant form"

Base.summary(a::WENO{N}) where N = string("WENO reconstruction order ", N*2-1, " in ", formulation(a))

Base.show(io::IO, a::WENO{N, FT, RX, RY, RZ, VI, WF, PP}) where {N, FT, RX, RY, RZ, VI, WF, PP} =
    print(io, summary(a), " \n",
              " Smoothness formulation: ", "\n",
              "    └── $(WF ? "Z-weno" : "JS-weno") $(VI<:SmoothnessStencil ? "using $VI" : "") \n",
              a.bounds isa Nothing ? "" : " Bounds : \n    └── $(a.bounds) \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.boundary_scheme) , "\n",
              " Symmetric scheme: ", "\n",
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
