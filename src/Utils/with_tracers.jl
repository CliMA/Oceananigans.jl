"""
    with_tracers(tracer_names, initial_tuple, tracer_default)

Create a tuple corresponding to the solution variables `u`, `v`, `w`,
and `tracer_names`. `initial_tuple` is a `NamedTuple` that at least has
fields `u`, `v`, and `w`, and may have some fields corresponding to
the names in `tracer_names`. `tracer_default` is a function that produces
a default tuple value for each tracer if not included in `initial_tuple`.
"""
@inline with_tracers(tracer_names, initial_tuple::NamedTuple, tracer_default; with_velocities=false) =
    with_tracers(tracer_names, initial_tuple::NamedTuple, tracer_default, with_velocities)

@inline function with_tracers(tracer_names::TN, initial_tuple::IT, tracer_default,
                              with_velocities) where {TN, IT<:NamedTuple}

    if with_velocities
        solution_values = (initial_tuple.u,
                           initial_tuple.v,
                           initial_tuple.w)

        solution_names = (:u, :v, :w)
    else
        solution_values = tuple()
        solution_names = tuple()
    end

    next = ntuple(Val(length(tracer_names))) do n
        Base.@_inline_meta
        name = tracer_names[n]
        if name ∈ propertynames(initial_tuple)
            getproperty(initial_tuple, name)
        else
            tracer_default(tracer_names, initial_tuple)
        end
    end

    solution_values = (solution_values..., next...)
    solution_names = (solution_names..., tracer_names...)

    return NamedTuple{solution_names}(solution_values)
end

# If the initial tuple is 'nothing', return nothing.
with_tracers(tracer_names, ::Nothing, args...; kwargs...) = nothing
