"""
$(TYPEDSIGNATURES)

Return the value of `a` at index `(i, j, k)`: `a` itself if it is a `Number` or `nothing`,
`a[i, j, k]` otherwise (an array or `Field`), and `a(i, j, k, grid, args...)` if it is a `Function`.

```jldoctest
using Oceananigans
using Oceananigans.Utils: getvalue

grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
c = CenterField(grid)
set!(c, 3)
f(i, j, k, grid, a) = a * k

(getvalue(7, 1, 1, 2, grid), getvalue(c, 1, 1, 2, grid), getvalue(f, 1, 1, 2, grid, 10))

# output

(7, 3.0, 20)
```
"""
@inline getvalue(::Nothing,        i, j, k, grid, args...) = nothing
@inline getvalue(a::Number,        i, j, k, grid, args...) = a
@inline getvalue(a,                i, j, k, grid, args...) = @inbounds a[i, j, k]
@inline getvalue(a::F,             i, j, k, grid, args...) where F<:Function = a(i, j, k, grid, args...)
