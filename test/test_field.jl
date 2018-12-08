"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.

# Examples
```julia-repl
julia> test_init_field((10, 10, 5), (10, 10, 10), CellField)
```
"""
function test_init_field(g, ftf)
    f = ftf(g)
    size(f) == size(g)
end

"""
    test_set_field(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::Field, v)`
function.

# Examples
```julia-repl
julia> test_init_field((10, 10, 5), (10, 10, 10), FaceFieldX, 1//7)
```
"""
function test_set_field(g, ftf, val)
    f = ftf(g)
    set!(f, val)
    f.data == val * ones(size(f))
end

function test_add_field(g, ftf, val1, val2)
    f1 = ftf(g)
    f2 = ftf(g)
    set!(u, val1)
    set!(v, val2)
    f3 = f1 + f2
    fans = (val1 + val2) * ones(N)
    f3.data == fans
end
