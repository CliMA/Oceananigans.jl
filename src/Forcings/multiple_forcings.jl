struct MultipleForcings{N, F}
    forcings :: F
end

"""
    MultipleForcings(forcings)

Return a lightweight tuple-wrapper representing multiple user-defined `forcings`.
Each forcing in `forcings` is added to the specified field's tendency.
"""
function MultipleForcings(forcings)
    N = length(forcings)
    F = typeof(forcings)
    return MultipleForcings{N, F}(forcings)
end

function regularize_forcing(forcing_tuple::Tuple, field, field_name, model_field_names)
    forcings = Tuple(regularize_forcing(f, field, field_name, model_field_names)
                     for f in forcing_tuple)
    return MultipleForcings(forcings)
end

# The magic
@inline function (mf::MultipleForcings{N})(i, j, k, grid, clock, model_fields) where N
    total_forcing = zero(eltype(grid))
    ntuple(Val(N)) do n
        Base.@_inline_meta
        nth_forcing = mf.forcings[n]
        total_forcing += nth_forcing(i, j, k, grid, clock, model_fields)
    end
    return total_forcing
end

