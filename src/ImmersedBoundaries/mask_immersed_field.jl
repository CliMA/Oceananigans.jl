using KernelAbstractions: @kernel, @index
using KernelAbstractions: NoneEvent
using Statistics
using Oceananigans.Architectures: architecture, device_event
using Oceananigans.Fields: location, ZReducedField, Field

instantiate(X) = X()

#####
##### Outer functions
#####

function mask_immersed_field!(field::Field, value=zero(field.grid); blocking=true)
    if blocking
        event = mask_immersed_field!(field, field.grid, location(field), value)
        wait(device(architecture(field)), event)
        return nothing
    else
        return mask_immersed_field!(field, field.grid, location(field), value)
    end
end

function mask_immersed_field_xy!(field value=zero(field.grid); k, blocking, mask=peripheral_node)
    if blocking
        event = mask_immersed_field_xy!(field, field.grid, location(field), value; k, mask)
        wait(device(architecture(field)), event)
        return nothing
    else
        return mask_immersed_field_xy!(field, field.grid, location(field), value; k, mask)
    end
end

#####
##### Implementations
#####

mask_immersed_field!(field, grid, loc, value) = NoneEvent()

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

mask_immersed_reduced_field_xy!(field,     args...; kw...) = NoneEvent()
mask_immersed_reduced_field_xy!(::Nothing, args...; kw...) = NoneEvent()
mask_immersed_field_xy!(field, value=zero(field.grid); k, mask=peripheral_node) =
    mask_immersed_field_xy!(field, field.grid, location(field), value; k, mask)

"""
    mask_immersed_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value; k, mask=peripheral_node)

Mask `field` on `grid` with a `value` on the slices `[:, :, k]` where `mask` is `true`.
"""
function mask_immersed_reduced_field_xy!(field, grid::ImmersedBoundaryGrid, loc, value; k, mask)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xy,
                   _mask_immersed_reduced_field_xy!, field, loc, grid, value, k, mask;
                   dependencies = device_event(arch))
end

@kernel function _mask_immersed_reduced_field_xy!(field, loc, grid, value, k, mask)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field, mask)
end

#####
##### mask_immersed_velocities for NonhydrostaticModel
#####

mask_immersed_velocities!(U, arch, grid) = tuple(NoneEvent())
mask_immersed_velocities!(U, arch, grid::ImmersedBoundaryGrid) = Tuple(mask_immersed_field!(q; blocking=false) for q in U)

#####
##### Masking for GridFittedBoundary
#####

@inline scalar_mask(i, j, k, grid, ::AbstractGridFittedBoundary, LX, LY, LZ, value, field, mask=peripheral_node) =
    @inbounds ifelse(mask(i, j, k, grid, LX, LY, LZ), value, field[i, j, k])

