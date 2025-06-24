import Oceananigans

#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, FT2, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}

    "Bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "Advection scheme used near boundaries"
    buffer_scheme :: CA
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function WENO{N, FT, FT2}(bounds::PP, buffer_scheme::CA,
                              advecting_velocity_scheme :: SI) where {N, FT, FT2, PP, CA, SI}

        return new{N, FT, FT2, PP, CA, SI}(bounds, buffer_scheme, advecting_velocity_scheme)
    end
end

"""
    WENO([FT=Float64, FT2=Float32;]
         order = 5,
         grid = nothing,
         bounds = nothing)

Construct a weighted essentially non-oscillatory advection scheme of order `order` with precision `FT`.

Arguments
=========

- `FT`: The floating point type used in the scheme. Default: `Oceananigans.defaults.FloatType`
- `FT2`: The floating point type used in some performance-critical parts of the scheme. Default: `Float32`

Keyword arguments
=================

- `order`: The order of the WENO advection scheme. Default: 5
- `grid`: (defaults to `nothing`)

Examples
========
```jldoctest
julia> using Oceananigans

julia> WENO()
WENO(order=5)
 Boundary scheme:
    └── WENO(order=3)
 Symmetric scheme:
    └── Centered(order=4)
```

```jldoctest
julia> using Oceananigans

julia> Nx, Nz = 16, 10;

julia> Lx, Lz = 1e4, 1e3;

julia> chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

julia> grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4), topology=(Periodic, Flat, Bounded),
                              x = (0, Lx), z = chebychev_spaced_z_faces);

julia> WENO(grid; order=7)
WENO(order=7)
 Boundary scheme:
    └── WENO(order=5)
 Symmetric scheme:
    └── Centered(order=6)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32;
              order = 5,
              grid = nothing,
              bounds = nothing)

    if !(grid isa Nothing)
        FT = eltype(grid)
    end

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if !isnothing(bounds)
        @warn "Bounds preserving WENO is experimental."
    end

    if order < 3
        # WENO(order=1) is equivalent to UpwindBiased(order=1)
        return UpwindBiased(FT; order=1)
    else
        N = Int((order + 1) ÷ 2)
        advecting_velocity_scheme = Centered(FT; grid, order = order - 1)
        buffer_scheme = WENO(FT; grid, order=order-2, bounds)
    end

    return WENO{N, FT, FT2}(bounds, buffer_scheme, advecting_velocity_scheme)
end

WENO(grid, FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32; kwargs...) = WENO(FT, FT2; grid, kwargs...)

# Flavours of WENO
const PositiveWENO = WENO{<:Any, <:Any, <:Any, <:Tuple}

Base.summary(a::WENO{N}) where N = string("WENO(order=", N*2-1, ")")

Base.show(io::IO, a::WENO{N, FT, FT2, PP}) where {N, FT, FT2, PP} =
    print(io, summary(a), " \n",
              a.bounds isa Nothing ? "" : " Bounds : \n    └── $(a.bounds) \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme) , "\n",
              " Symmetric scheme: ", "\n",
              "    └── ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
     WENO{N, FT, FT2}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.buffer_scheme),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
    WENO{N, FT, FT2}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.buffer_scheme),
                     on_architecture(to, scheme.advecting_velocity_scheme))