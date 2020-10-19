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
