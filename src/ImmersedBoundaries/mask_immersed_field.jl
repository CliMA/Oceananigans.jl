using KernelAbstractions: @kernel, @index
using KernelAbstractions: NoneEvent
using Statistics
using Oceananigans.Architectures: architecture, device_event
using Oceananigans.Fields: location, ZReducedField, Field

instantiate(X) = X()

mask_immersed_field!(field, grid, loc, value) = NoneEvent()
mask_immersed_field!(field::Field, value=zero(eltype(field.grid))) =
    mask_immersed_field!(field, field.grid, location(field), value)

"""
    mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value)

masks `field` defined on `grid` with a value `val` at locations where `peripheral_node` evaluates to `true`
"""
function mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, value; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

mask_immersed_reduced_field_xy!(field, args...; kw...) = NoneEvent()
mask_immersed_reduced_field_xy!(field, value=zero(eltype(field.grid)); k, immersed_function = peripheral_node) =
    mask_immersed_reduced_field_xy!(field, field.grid, location(field), value; k, immersed_function)

"""
    mask_immersed_reduced_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value; k, immersed_function = peripheral_node)

masks a `field` defined on `grid` with a value `val` at locations in the plane `[:, :, k]` where `immersed_function` evaluates to `true`
"""
function mask_immersed_reduced_field_xy!(field, grid::ImmersedBoundaryGrid, loc, value; k, immersed_function)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xy,
                   _mask_immersed_reduced_field_xy!, field, loc, grid, value, k, immersed_function;
                   dependencies = device_event(arch))
end

@kernel function _mask_immersed_reduced_field_xy!(field, loc, grid, value, k, immersed_function)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field, immersed_function)
end

#####
##### mask_immersed_velocities for NonhydrostaticModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())

#####
##### Masking for GridFittedBoundary
#####

@inline function scalar_mask(i, j, k, grid, ::AbstractGridFittedBoundary, LX, LY, LZ, value, field, immersed_function = peripheral_node)
    return @inbounds ifelse(immersed_function(i, j, k, grid, LX, LY, LZ),
                            value,
                            field[i, j, k])
end

mask_immersed_velocities!(U, arch, grid::ImmersedBoundaryGrid) = Tuple(mask_immersed_field!(q) for q in U)
