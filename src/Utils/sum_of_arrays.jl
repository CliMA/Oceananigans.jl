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

# Convenience constructors for velocities with components (u, v, w) that
# throw away the `nothing` values. We assume that we pass at least one valid velocity field.

const NT = NamedTuple

# 2 - tuple Valid velocity fields
@inline sum_of_velocities(U1::NT, U2::NT) = (u = SumOfArrays{2}(U1.u, U2.u),
                                             v = SumOfArrays{2}(U1.v, U2.v),
                                             w = SumOfArrays{2}(U1.w, U2.w))

# 2 - tuple Combinations with `nothing`
@inline sum_of_velocities(U1::NT, ::Nothing) = U1
@inline sum_of_velocities(::Nothing, U2::NT) = U2

# 3 - tuple Valid velocity fields
@inline sum_of_velocities(U1::NT, U2::NT, U3::NT) = (u = SumOfArrays{3}(U1.u, U2.u, U3.u),
                                                     v = SumOfArrays{3}(U1.v, U2.v, U3.v),
                                                     w = SumOfArrays{3}(U1.w, U2.w, U3.w))

# 3 - tuple Combinations with `nothing`
@inline sum_of_velocities(U1::NT, U2::NT, ::Nothing) = sum_of_velocities(U1, U2)
@inline sum_of_velocities(U1::NT, ::Nothing, U3::NT) = sum_of_velocities(U1, U3)
@inline sum_of_velocities(::Nothing, U2::NT, U3::NT) = sum_of_velocities(U2, U3)

@inline sum_of_velocities(U1::NT, ::Nothing, ::Nothing) = U1
@inline sum_of_velocities(::Nothing, U2::NT, ::Nothing) = U2
@inline sum_of_velocities(::Nothing, ::Nothing, U3::NT) = U3