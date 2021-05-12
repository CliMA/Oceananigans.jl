using KernelAbstractions

mask_immersed_field!(field, ib) = NoneEvent()

mask_immersed_field!(field::AbstractField{X, Y, Z}, ib::GridFittedImmersedBoundary) where {X, Y, Z} =
    launch!(architecture(field), grid, :xyz, _mask_immersed_field!, field, (X, Y, Z), ib, dependencies = barrier)

@kernel function _mask_immersed_field(field, loc, ib)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = ifelse(solid_node(loc..., i, j, k, grid, ib), 0, field[i, j, k])
end
