using KernelAbstractions
using Statistics
using Oceananigans.Architectures: architecture, device_event
using Oceananigans.Fields: location

mask_immersed_field!(field::AbstractField, loc=location(field)) = mask_immersed_field!(field, field.grid, loc)
mask_immersed_field!(field, grid, loc) = NoneEvent()
instantiate(X) = X()

function mask_immersed_field!(field, grid::ImmersedBoundaryGrid, loc)
    arch = architecture(field)
    loc = instantiate.(loc)
    mask_value = zero(eltype(grid))
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, mask_value; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

#####
##### mask_immersed_velocities for IncompressibleModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())
