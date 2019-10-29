"""
    FieldMaximum(mapping, field)

An object for calculating the maximum of a `mapping` function applied
element-wise to `field`.

Examples
=======
```julia
julia> model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)));

julia> max_abs_u = FieldMaximum(abs, model.velocities.u);

julia> max_w² = FieldMaximum(x->x^2, model.velocities.w);
```
"""
struct FieldMaximum{F, M}
    mapping :: M
      field :: F
end

(m::FieldMaximum)(args...) = maximum(m.mapping, m.field.data.parent)

(m::FieldMaximum{<:NamedTuple})(args...) =
    NamedTuple{propertynames(m.field)}(maximum(m.mapping, f.data.parent) for f in m.field)
