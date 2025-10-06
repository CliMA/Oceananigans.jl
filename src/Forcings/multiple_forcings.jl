using Adapt

struct MultipleForcings{N, F}
    forcings :: F
end

Adapt.adapt_structure(to, mf::MultipleForcings) = MultipleForcings(adapt(to, mf.forcings))
on_architecture(to, mf::MultipleForcings) = MultipleForcings(on_architecture(to, mf.forcings))

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

MultipleForcings(args...) = MultipleForcings(tuple(args...))

function regularize_forcing(forcing_tuple::Tuple, field, field_name, model_field_names)
    forcings = Tuple(regularize_forcing(f, field, field_name, model_field_names)
                     for f in forcing_tuple)
    return MultipleForcings(forcings)
end

regularize_forcing(mf::MultipleForcings, args...) = regularize_forcing(mf.forcings, args...)

@inline (mf::MultipleForcings{1})(i, j, k, grid, clock, model_fields) = mf.forcings[1](i, j, k, grid, clock, model_fields)

@inline (mf::MultipleForcings{2})(i, j, k, grid, clock, model_fields) = mf.forcings[1](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[2](i, j, k, grid, clock, model_fields)

@inline (mf::MultipleForcings{3})(i, j, k, grid, clock, model_fields) = mf.forcings[1](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[2](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[3](i, j, k, grid, clock, model_fields)

@inline (mf::MultipleForcings{4})(i, j, k, grid, clock, model_fields) = mf.forcings[1](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[2](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[3](i, j, k, grid, clock, model_fields) +
                                                                        mf.forcings[4](i, j, k, grid, clock, model_fields)

@generated function (mf::MultipleForcings{N})(i, j, k, grid, clock, model_fields) where N
    quote
        total_forcing = zero(grid)
        forcings = mf.forcings
        Base.@_inline_meta
        $([:(@inbounds total_forcing += forcings[$n](i, j, k, grid, clock, model_fields)) for n in 1:N]...)
        return total_forcing
    end
end

Base.summary(mf::MultipleForcings) = string("MultipleForcings with ", length(mf.forcings), " forcing",
                                            ifelse(length(mf.forcings) > 1, "s", ""))

function Base.show(io::IO, mf::MultipleForcings)
    start = summary(mf) * ":"

    Nf = length(mf.forcings)
    if Nf > 1
        body = [string("├ ", prettysummary(f), "\n") for f in mf.forcings[1:end-1]]
    else
        body = []
    end

    push!(body, string("└ ", prettysummary(mf.forcings[end])))

    print(io, start, "\n", body...)

    return nothing
end

