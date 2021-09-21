using KernelAbstractions
using Statistics
using Oceananigans.Architectures: architecture, device_event
using Oceananigans.Fields: location, AbstractDataField

instantiate(X) = X()

mask_immersed_field!(field::AbstractField, value=zero(eltype(field.grid))) = mask_immersed_field!(field, field.grid, location(field), value)
mask_immersed_field!(field, grid, loc, value) = NoneEvent()

function mask_immersed_field!(field::AbstractDataField, grid::ImmersedBoundaryGrid, loc, value)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, value; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

#####
##### mask_immersed_velocities for NonhydrostaticModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())
