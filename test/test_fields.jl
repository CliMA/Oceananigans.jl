"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.

# Examples
```julia-repl
julia> test_init_field((10, 10, 5), (10, 10, 10), CellField)
```
"""
function test_init_field(mm::ModelMetadata, g::Grid, field_type)
    f = field_type(mm, g, mm.float_type)
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
function test_set_field(mm::ModelMetadata, g::Grid, field_type, val::Number)
    f = field_type(mm, g, mm.float_type)
    set!(f, val)
    f.data ≈ val * ones(size(f))
end

function test_add_field(mm::ModelMetadata, g::Grid, field_type, val1::Number, val2::Number)
    f1 = field_type(mm, g, mm.float_type)
    f2 = field_type(mm, g, mm.float_type)

    set!(f1, val1)
    set!(f2, val2)
    f3 = f1 + f2

    val3 = convert(mm.float_type, val1) + convert(mm.float_type, val2)
    f_ans = val3 * ones(size(f1))
    f3.data ≈ f_ans
end
