using KernelAbstractions
using Statistics
using Oceananigans.Architectures: architecture, device_event
using Oceananigans.Fields: location, AbstractReducedField

instantiate(X) = X()

mask_immersed_field!(field::AbstractDataField, value=zero(eltype(field.grid))) =
    mask_immersed_field!(field, field.grid, location(field), value)

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

mask_immersed_reduced_field_xy!(field::AbstractReducedField, value=zero(eltype(field.grid)); k) =
    mask_immersed_reduced_field_xy!(field, field.grid, location(field), value; k)

mask_immersed_reduced_field_xy!(field, grid, loc, value; k) = NoneEvent()

function mask_immersed_reduced_field_xy!(field::AbstractReducedField, grid::ImmersedBoundaryGrid, loc, value; k)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xy, _mask_immersed_reduced_field_xy!, field, loc, grid, value, k; dependencies = device_event(arch))
end

@kernel function _mask_immersed_reduced_field_xy!(field, loc, grid, value, k)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, 1] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

#####
##### mask_immersed_velocities for NonhydrostaticModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())
