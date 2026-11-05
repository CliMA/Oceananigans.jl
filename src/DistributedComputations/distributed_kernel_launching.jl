using Oceananigans.Utils: _launch!

@inline function Utils._launch!(arch::Distributed, args...; kwargs...)
    child_arch = child_architecture(arch)
    return _launch!(child_arch, args...; kwargs...)
end
