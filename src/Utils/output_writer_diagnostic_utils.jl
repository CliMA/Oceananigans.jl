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
    validate_interval(frequency, interval)

Ensure that frequency and interval are not both `nothing`.
"""
function validate_interval(frequency, interval)
    isnothing(frequency) && isnothing(interval) && @error "Must specify a frequency or interval!"
    return nothing
end

has_interval(obj) = :interval in propertynames(obj) && obj.interval != nothing
has_frequency(obj) = :frequency in propertynames(obj) && obj.frequency != nothing

function interval_is_ripe(clock, obj)
    if has_interval(obj) && clock.time >= obj.previous + obj.interval
        obj.previous = clock.time - rem(clock.time, obj.interval)
        return true
    else
        return false
    end
end

frequency_is_ripe(clock, obj) = has_frequency(obj) && clock.iteration % obj.frequency == 0

function time_to_run(clock, output_writer)

    interval_is_ripe(clock, output_writer) && return true
    frequency_is_ripe(clock, output_writer) && return true

    # If the output writer does not specify an interval or frequency,
    # it is unable to write output and we throw an error as a convenience.
    has_interval(output_writer) || has_frequency(output_writer) ||
        error("$(typeof(output_writer)) must have a frequency or interval specified!")

    return false
end
