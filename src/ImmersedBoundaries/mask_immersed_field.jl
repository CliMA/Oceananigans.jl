using KernelAbstractions
using Oceananigans.Architectures: architecture, device_event

mask_immersed_field!(field, ib) = NoneEvent()

function mask_immersed_field!(field::AbstractField{X, Y, Z}, ib::GridFittedImmersedBoundary) where {X, Y, Z}
    arch = architecture(field)
    loc = (X(), Y(), Z())
    return launch!(arch, field.grid, :xyz, _mask_immersed_field!, field, loc, ib; dependencies = device_event(arch))
end

@kernel function _mask_immersed_field!(field, loc, ib)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = ifelse(solid_node(loc..., i, j, k, ib.grid, ib), 0, field[i, j, k])
end
