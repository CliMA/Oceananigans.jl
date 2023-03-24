using Base: @propagate_inbounds

import Base: getindex

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
@propagate_inbounds getindex(s::SumOfArrays{3}, i...) = getindex(s.arrays[1], i...) + getindex(s.arrays[2], i...) + getindex(s.arrays[3], i...)