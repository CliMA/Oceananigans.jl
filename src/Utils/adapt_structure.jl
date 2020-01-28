import Adapt
using OffsetArrays

# Adapt an offset CuArray to work nicely with CUDA kernels.
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(Adapt.adapt(to, parent(x)), x.offsets)

# Need to adapt SubArray indices as well.
# See: https://github.com/JuliaGPU/Adapt.jl/issues/16
#Adapt.adapt_structure(to, A::SubArray{<:Any,<:Any,AT}) where {AT} =
#    SubArray(adapt(to, parent(A)), adapt.(Ref(to), parentindices(A)))
