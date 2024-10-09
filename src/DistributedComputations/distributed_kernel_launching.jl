import Oceananigans.Utils: launch!

function launch!(arch::Distributed, args...; kwargs...)
    child_arch = child_architecture(arch)
    return launch!(child_arch, args...; kwargs...)
end

# Disambiguiate
@inline function launch!(arch::Distributed, grid, workspec_tuple::Tuple, args...; kwargs...)
    child_arch = child_architecture(arch)
    return launch!(child_arch, grid, workspec_tuple, args...; kwargs...)
end

