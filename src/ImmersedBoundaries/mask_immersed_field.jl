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
    @inbounds field[i, j, k] = ifelse(solid_node(loc..., i, j, k, grid), 0, field[i, j, k])
end
