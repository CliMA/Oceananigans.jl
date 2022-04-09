using Adapt

struct MultipleForcings{N, F}
    forcings :: F
end

Adapt.adapt_structure(to, mf::MultipleForcings) = MultipleForcings(adapt(to, mf.forcings))

Base.getindex(mf::MultipleForcings, i) = mf.forcings[i]

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
    forcings = mf.forcings
    ntuple(Val(N)) do n
        Base.@_inline_meta
        @inbounds begin
            nth_forcing = forcings[n]
            total_forcing += nth_forcing(i, j, k, grid, clock, model_fields)
        end
    end
    return total_forcing
end

Base.summary(mf::MultipleForcings) = string("MultipleForcings with ", length(mf.forcings), " forcing",
                                            ifelse(length(mf.forcings) > 1, "s", ""))

function Base.show(io::IO, mf::MultipleForcings)
    start = summary(mf) * ":"

    Nf = length(mf.forcings)
    if Nf > 1
        body = [string("├ ", prettysummary(f), '\n') for f in mf.forcings[1:end-1]]
    else
        body = []
    end

    push!(body, string("└ ", prettysummary(mf.forcings[end])))

    print(io, start, '\n', body...)

    return nothing
end

