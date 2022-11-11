#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, XT, YT, ZT, WF, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    
    "Coefficient for ENO reconstruction on x-faces" 
    coeff_xᶠᵃᵃ::XT
    "Coefficient for ENO reconstruction on x-centers"
    coeff_xᶜᵃᵃ::XT
    "Coefficient for ENO reconstruction on y-faces"
    coeff_yᵃᶠᵃ::YT
    "Coefficient for ENO reconstruction on y-centers"
    coeff_yᵃᶜᵃ::YT
    "Coefficient for ENO reconstruction on z-faces"
    coeff_zᵃᵃᶠ::ZT
    "Coefficient for ENO reconstruction on z-centers"
    coeff_zᵃᵃᶜ::ZT

    "Bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "Advection scheme used near boundaries"
    buffer_scheme :: CA
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function WENO{N, FT, WF}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                             coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                             coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                             bounds::PP, buffer_scheme::CA,
                             advecting_velocity_scheme :: SI) where {N, FT, XT, YT, ZT, WF, PP, CA, SI}

            return new{N, FT, XT, YT, ZT, WF, PP, CA, SI}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                                          coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                                          coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                                          bounds, buffer_scheme, advecting_velocity_scheme)
    end
end

"""
    WENO([FT=Float64;] 
         order = 5,
         grid = nothing, 
         zweno = true,
         bounds = nothing)
               
Construct a weigthed essentially non-oscillatory advection scheme of order `order`.

Keyword arguments
=================

- `order`: The order of the WENO advection scheme. Default: 5
- `grid`: (defaults to `nothing`)
- `vector_invariant`: The stencil for which the vector-invariant form of the advection
                      scheme would use. Options `VelocityStencil()` or `VorticityStencil()`;
                      defaults to `nothing`.

- `zweno`: When `true` implement a Z-WENO formulation for the WENO weights calculation.
           (defaults to `true`)

Examples
========
```jldoctest
julia> using Oceananigans;

julia> WENO()
WENO reconstruction order 5 
 Smoothness formulation: 
    └── Z-weno  
 Boundary scheme: 
    └── WENO reconstruction order 3 
 Symmetric scheme: 
    └── Centered reconstruction order 4
 Directions:
    ├── X regular 
    ├── Y regular 
    └── Z regular
```

```jldoctest
julia> using Oceananigans;

julia> Nx, Nz = 16, 10;

julia> Lx, Lz = 1e4, 1e3;

julia> chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

julia> grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4), topology=(Periodic, Flat, Bounded),
                              x = (0, Lx), z = chebychev_spaced_z_faces);

julia> WENO(grid; order=7)
WENO reconstruction order 7 in Flux form 
 Smoothness formulation: 
    └── Z-weno  
 Boundary scheme: 
    └── WENO reconstruction order 5 in Flux form
 Symmetric scheme: 
    └── Centered reconstruction order 6
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
        buffer_scheme   = WENO(FT; grid, order = order - 2, zweno, vector_invariant, bounds)
        advecting_velocity_scheme = Centered(FT; grid, order = order - 1)
    end

    return WENO{N, FT, zweno}(weno_coefficients..., bounds, buffer_scheme, advecting_velocity_scheme)
end

WENO(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid, kwargs...)

# Some usefull aliases
WENOThirdOrder(grid=nothing, FT::DataType=Float64;  kwargs...) = WENO(grid, FT; order=3, kwargs...)
WENOFifthOrder(grid=nothing, FT::DataType=Float64;  kwargs...) = WENO(grid, FT; order=5, kwargs...)

# Flavours of WENO
const ZWENO        = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}
const PositiveWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple}

Base.summary(a::WENO{N}) where N = string("WENO reconstruction order ", N*2-1, " in ", formulation(a))

Base.show(io::IO, a::WENO{N, FT, RX, RY, RZ, VI, WF, PP}) where {N, FT, RX, RY, RZ, VI, WF, PP} =
    print(io, summary(a), " \n",
              " Smoothness formulation: ", "\n",
              "    └── $(WF ? "Z-weno" : "JS-weno") \n",
              a.bounds isa Nothing ? "" : " Bounds : \n    └── $(a.bounds) \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme) , "\n",
              " Symmetric scheme: ", "\n",
              "    └── ", summary(a.advecting_velocity_scheme) , "\n",
              " Directions:", "\n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )

Adapt.adapt_structure(to, scheme::WENO{N, FT, XT, YT, ZT, WF, PP}) where {N, FT, XT, YT, ZT, WF, PP} =
     WENO{N, FT, WF}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                     Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                     Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                     Adapt.adapt(to, scheme.bounds),
                     Adapt.adapt(to, scheme.buffer_scheme),
                     Adapt.adapt(to, scheme.advecting_velocity_scheme))

# Retrieve precomputed coefficients (+2 for julia's 1 based indices)
@inline retrieve_coeff(scheme::WENO, r, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[r+2][i] 
@inline retrieve_coeff(scheme::WENO, r, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[r+2][i] 
