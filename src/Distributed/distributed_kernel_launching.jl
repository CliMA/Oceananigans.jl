import Oceananigans.Utils: launch!, heuristic_workgroup

function launch!(arch::AbstractMultiArchitecture, args...; kwargs...)
    child_arch = child_architecture(arch)
    return launch!(child_arch, args...; kwargs...)
end

heuristic_workgroup(arch::AbstractMultiArchitecture, Nx, Ny, Nz) = heuristic_workgroup(child_architecture(arch), Nx, Ny, Nz)
