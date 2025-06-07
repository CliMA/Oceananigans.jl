import Base: push!, getindex, setindex!

using OrderedCollections: OrderedDict
using Oceananigans: AbstractOutputWriter, AbstractDiagnostic

#####
##### Utilities shared between diagnostics and output writers
#####

defaultname(::AbstractDiagnostic, nelems) = Symbol(:diag, nelems+1)

defaultname(::AbstractOutputWriter, nelems) = Symbol(:writer, nelems+1)

const DiagOrWriterDict = OrderedDict{S, <:Union{AbstractDiagnostic, AbstractOutputWriter}} where S

push!(container::DiagOrWriterDict, elem) = (container[defaultname(elem, length(container))] = elem; nothing)

getindex(container::DiagOrWriterDict, inds::Integer...) = getindex(container.vals, inds...)

setindex!(container::DiagOrWriterDict, newvals, inds::Integer...) = setindex!(container.vals, newvals, inds...)

push!(container::DiagOrWriterDict, elems...) = (foreach(e -> push!(container, e), elems); nothing)

