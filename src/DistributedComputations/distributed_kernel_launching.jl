import Oceananigans.Utils: launch!

function launch!(arch::Distributed, args...; kwargs...)
    child_arch = child_architecture(arch)
    return launch!(child_arch, args...; kwargs...)
end

