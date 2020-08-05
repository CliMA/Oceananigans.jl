import Base: push!, getindex, setindex!

using OrderedCollections: OrderedDict
using Oceananigans: AbstractOutputWriter, AbstractDiagnostic

#####
##### Utilities shared between diagnostics and output writers
#####

defaultname(::AbstractDiagnostic, nelems) = Symbol(:diag, nelems+1)
defaultname(::AbstractOutputWriter, nelems) = Symbol(:writer, nelems+1)

const DiagOrWriterDict = OrderedDict{S, <:Union{AbstractDiagnostic, AbstractOutputWriter}} where S

function push!(container::DiagOrWriterDict, elem)
    name = defaultname(elem, length(container))
    container[name] = elem
    return nothing
end

getindex(container::DiagOrWriterDict, inds::Integer...) = getindex(container.vals, inds...)
setindex!(container::DiagOrWriterDict, newvals, inds::Integer...) = setindex!(container.vals, newvals, inds...)

function push!(container::DiagOrWriterDict, elems...)
    for elem in elems
        push!(container, elem)
    end
    return nothing
end

"""
    validate_intervals(iteration_interval, time_interval)

Warn the user if iteration and time intervals are both `nothing`.
"""
function validate_intervals(iteration_interval, time_interval)
    isnothing(iteration_interval) && isnothing(time_interval) && @warn "You have not specified any intervals."
    return nothing
end

has_iteration_interval(obj) = :iteration_interval in propertynames(obj) && obj.iteration_interval != nothing
has_time_interval(obj) = :time_interval in propertynames(obj) && obj.time_interval != nothing

iteration_interval_is_ripe(clock, obj) = has_iteration_interval(obj) && clock.iteration % obj.iteration_interval == 0

function time_interval_is_ripe(clock, obj)
    if has_time_interval(obj) && clock.time >= obj.previous + obj.time_interval
        obj.previous = clock.time - rem(clock.time, obj.time_interval)
        return true
    else
        return false
    end
end

function time_to_run(clock, diag)
    iteration_interval_is_ripe(clock, diag) && return true
    has_time_interval(clock, diag) && return true

    # If the output writer does not specify any intervals,
    # it is unable to write output and we throw an error as a convenience.
    iteration_interval_is_ripe(diag) || has_time_interval(diag) ||
        error("$(typeof(diag)) must have an iteration or time interval specified!")

    return false
end
