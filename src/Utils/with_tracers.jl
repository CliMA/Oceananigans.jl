"""
    with_tracers(tracer_names, initial_tuple::NamedTuple, tracer_default; with_velocities=false)

Create a tuple corresponding to the solution variables `u`, `v`, `w`,
and `tracer_names`. `initial_tuple` is a `NamedTuple` that at least has
fields `u`, `v`, and `w`, and may have some fields corresponding to
the names in `tracer_names`. `tracer_default` is a function that produces
a default tuple value for each tracer if not included in `initial_tuple`.
"""
function with_tracers(tracer_names, initial_tuple::NamedTuple, tracer_default; with_velocities=false)
    solution_values = [] # Array{Any, 1}
    solution_names = []

    if with_velocities
        push!(solution_values, initial_tuple.u)
        push!(solution_values, initial_tuple.v)
        push!(solution_values, initial_tuple.w)

        append!(solution_names, [:u, :v, :w])
    end

    for name in tracer_names
        tracer_elem = name ∈ propertynames(initial_tuple) ?
                         getproperty(initial_tuple, name) :
                         tracer_default(tracer_names, initial_tuple)

        push!(solution_values, tracer_elem)
    end

    append!(solution_names, tracer_names)

    return NamedTuple{Tuple(solution_names)}(Tuple(solution_values))
end

# If the initial tuple is 'nothing', return nothing.
with_tracers(tracer_names, ::Nothing, args...; kwargs...) = nothing
