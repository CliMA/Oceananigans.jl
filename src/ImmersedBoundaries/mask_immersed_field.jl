using KernelAbstractions
using Oceananigans.Architectures: architecture, device_event

mask_immersed_field!(field) = mask_immersed_field!(field, field.grid)
mask_immersed_field!(field, grid) = NoneEvent()

function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, grid::ImmersedBoundaryGrid) where {LX, LY, LZ}
    arch = architecture(field)
    loc = (LX(), LY(), LZ())
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., field)
end

#####
##### mask_immersed_velocities for IncompressibleModel
#####

mask_immersed_velocities!(U, arch, grid) = NoneEvent()

