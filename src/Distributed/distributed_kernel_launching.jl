import Oceananigans.Utils: launch!

function launch!(arch::MultiProcess, args...; kwargs...)
    child_arch = child_architecture(arch)
    return launch!(child_arch, args...; kwargs...)
end

