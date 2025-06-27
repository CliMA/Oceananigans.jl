import Oceananigans.Utils: _launch!

function _launch!(arch::Distributed, args...; kwargs...)
    child_arch = child_architecture(arch)
    return _launch!(child_arch, args...; kwargs...)
end

