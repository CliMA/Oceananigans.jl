using KernelAbstractions
using Statistics
using Oceananigans.Architectures: architecture, device_event

mask_immersed_field!(field) = mask_immersed_field!(field, field.grid)
mask_immersed_height_field!(field) = mask_immersed_height_field!(field, field.grid)

mask_immersed_field!(field, grid) = NoneEvent()
mask_immersed_height_field!(field, grid) = NoneEvent()

function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, grid::ImmersedBoundaryGrid) where {LX, LY, LZ}
    arch = architecture(field)
    loc = (LX(), LY(), LZ())
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, 0; dependencies = device_event(arch))
end

function mask_immersed_height_field!(h::AbstractField{LX, LY, LZ}, grid::ImmersedBoundaryGrid) where {LX, LY, LZ}
    arch = architecture(h)
    loc = (LX(), LY(), LZ())
    h₀ = mean(h)
    return launch!(arch, grid, :xyz, _mask_immersed_field!, h, loc, grid, h₀; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

#####
##### mask_immersed_velocities for IncompressibleModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())

