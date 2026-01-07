import Base: push!, getindex, setindex!

using OrderedCollections: OrderedDict
using Oceananigans: AbstractOutputWriter, AbstractDiagnostic

#####
##### Utilities shared between diagnostics and output writers
#####

getindex(container::DiagOrWriterDict, inds::Integer...) = getindex(container.vals, inds...)

setindex!(container::DiagOrWriterDict, newvals, inds::Integer...) = setindex!(container.vals, newvals, inds...)

function push!(container::DiagOrWriterDict, elems...)
    foreach(elem -> push!(container, elem), elems)
    return nothing
end

