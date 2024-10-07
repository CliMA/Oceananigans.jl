using Base: @propagate_inbounds

import Adapt: adapt_structure
import Base: getindex

"""
    SumOfArrays{N, F}

`SumOfArrays` objects hold `N` arrays/fields and return their sum when indexed.
"""
struct SumOfArrays{N, F}
    arrays :: F
    SumOfArrays{N}(arrays...) where N = new{N, typeof(arrays)}(arrays)
end

@propagate_inbounds function getindex(s::SumOfArrays{N}, i...) where N
    first = getindex(SumOfArrays{3}(s.arrays[1], s.arrays[2], s.arrays[3]), i...)
    last = getindex(SumOfArrays{N - 3}(s.arrays[4:N]...), i...)
    return first + last
end

@propagate_inbounds getindex(s::SumOfArrays{1}, i...) = getindex(s.arrays[1], i...)
@propagate_inbounds getindex(s::SumOfArrays{2}, i...) = getindex(s.arrays[1], i...) + getindex(s.arrays[2], i...)

@propagate_inbounds getindex(s::SumOfArrays{3}, i...) = 
    getindex(s.arrays[1], i...) + getindex(s.arrays[2], i...) + getindex(s.arrays[3], i...)
    
@propagate_inbounds getindex(s::SumOfArrays{4}, i...) = 
    getindex(s.arrays[1], i...) + getindex(s.arrays[2], i...) + getindex(s.arrays[3], i...) + getindex(s.arrays[4], i...)

adapt_structure(to, sum::SumOfArrays{N}) where N = SumOfArrays{N}((adapt_structure(to, array) for array in sum.arrays)...)