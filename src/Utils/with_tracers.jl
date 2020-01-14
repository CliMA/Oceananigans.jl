"""
    with_tracers(tracers, initial_tuple, tracer_default)

Create a tuple corresponding to the solution variables `u`, `v`, `w`,
and `tracers`. `initial_tuple` is a `NamedTuple` that at least has
fields `u`, `v`, and `w`, and may have some fields corresponding to
the names in `tracers`. `tracer_default` is a function that produces
a default tuple value for each tracer if not included in `initial_tuple`.
"""
function with_tracers(tracers, initial_tuple::NamedTuple, tracer_default; with_velocities=false)
    solution_values = [] # Array{Any, 1}
    solution_names = []

    if with_velocities
        push!(solution_values, initial_tuple.u)
        push!(solution_values, initial_tuple.v)
        push!(solution_values, initial_tuple.w)

        append!(solution_names, [:u, :v, :w])
    end

    for name in tracers
        tracer_elem = name ∈ propertynames(initial_tuple) ?
                        getproperty(initial_tuple, name) :
                        tracer_default(tracers, initial_tuple)

        push!(solution_values, tracer_elem)
    end

    append!(solution_names, tracers)

    return NamedTuple{Tuple(solution_names)}(Tuple(solution_values))
end
