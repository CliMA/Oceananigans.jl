using Base: @propagate_inbounds
using Oceananigans.Fields: AbstractField, location

import Adapt: adapt_structure
import Base: getindex, show, summary
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!

"""
    SumOfFields{N, F}

`SumOfFields` objects hold `N` field/fields and return their sum when indexed.
"""
struct SumOfFields{N, LX, LY, LZ, G, T, F} <: AbstractField{LX, LY, LZ, G, T, 3}
    fields :: F
      grid :: G
    
    function SumOfFields{N}(fields...) where N
        grid = first(fields).grid
        loc = location(first(fields))

        all(f.grid == grid for f in fields) || 
            throw(ArgumentError("All `fields` in `SumOfFields` must be on the same grid"))

        all(location(f) == loc for f in fields) || 
            throw(ArgumentError("All `fields` in `SumOfFields` must be on the same location"))

        T = eltype(first(fields).data)
        F = typeof(fields)
        G = typeof(grid)

        return new{N, loc..., G, T, F}(fields, grid)
    end
end

adapt_structure(to, sum::SumOfFields{N}) where N = SumOfFields{N}((adapt_structure(to, field) for field in sum.fields)...)

summary(::SumOfFields{N, LX, LY, LZ}) where {N, LX, LY, LZ}  = "Sum of $N fields at ($LX, $LY, $LZ)"

show(io::IO, sof::SumOfFields{N, LX, LY, LZ}) where {N, LX, LY, LZ} = print(io, summary(sof))

@propagate_inbounds function getindex(s::SumOfFields{N, LX, LY, LZ, G, T, F}, i...) where {N, LX, LY, LZ, G, T, F}
    first = getindex(SumOfFields{3, F, LX, LY, LZ, G, T}((s.fields[1], s.fields[2], s.fields[3]), s.grid), i...)
    last = getindex(SumOfFields{N - 3, F, LX, LY, LZ, G, T}(s.fields[4:N], s.grid), i...)
    return first + last
end

@propagate_inbounds getindex(s::SumOfFields{1}, i...) = getindex(s.fields[1], i...)
@propagate_inbounds getindex(s::SumOfFields{2}, i...) = getindex(s.fields[1], i...) + getindex(s.fields[2], i...)

@propagate_inbounds getindex(s::SumOfFields{3}, i...) = 
    getindex(s.fields[1], i...) + getindex(s.fields[2], i...) + getindex(s.fields[3], i...)
    
@propagate_inbounds getindex(s::SumOfFields{4}, i...) = 
    getindex(s.fields[1], i...) + getindex(s.fields[2], i...) + getindex(s.fields[3], i...) + getindex(s.fields[4], i...)

function mask_immersed_field!(sof::SumOfFields, value=zero(eltype(sof.grid)))
    for field in sof.fields
        mask_immersed_field!(field, value)
    end
    
    return nothing
end

function mask_immersed_field_xy!(sof::SumOfFields, value=zero(eltype(sof.grid)))
    for field in sof.fields
        mask_immersed_field_xy!(field, value)
    end
    
    return nothing
end

function fill_halo_regions!(sof::SumOfFields)
    for f in sof.fields
        fill_halo_regions!(f)
    end
    
    return nothing
end