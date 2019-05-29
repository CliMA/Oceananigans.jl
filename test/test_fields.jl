"""
    test_init_field(N, L, ftf)

Test that the field initialized by the field type function `ftf` on the grid g
has the correct size.
"""
function correct_field_size(arch::Architecture, g::Grid, field_type)
    f = field_type(arch, g)
    size(f) == size(g)
end

"""
    test_set_field(N, L, ftf, val)

Test that the field initialized by the field type function `ftf` on the grid g
can be correctly filled with the value `val` using the `set!(f::Field, v)`
function.
"""
function correct_field_value_was_set(arch::Architecture, g::Grid, field_type, val::Number)
    f = field_type(arch, g)
    set!(f, val)
    data(f) ≈ val * ones(size(f))
end

function correct_field_addition(arch::Architecture, g::Grid, field_type, val1::Number, val2::Number)
    f1 = field_type(arch, g)
    f2 = field_type(arch, g)

    set!(f1, val1)
    set!(f2, val2)
    f3 = f1 + f2

    val3 = convert(mm.float_type, val1) + convert(mm.float_type, val2)
    f_ans = val3 * ones(size(f1))
    f3.data ≈ f_ans
end
