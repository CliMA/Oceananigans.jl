# tracersomponents of the Redi rotation tensor

@inline function isopycnal_rotation_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    bz = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return ifelse(bz == 0, zero(FT), - bx / bz)
end

@inline function isopycnal_rotation_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(bz == 0, zero(FT), - bx / bz)
end

@inline function isopycnal_rotation_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    bz = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return ifelse(bz == 0, zero(FT), - by / bz)
end

@inline function isopycnal_rotation_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers) where FT
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(bz == 0, zero(FT), - by / bz)
end

@inline function isopycnal_rotation_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(bz == 0, zero(FT), (bx^2 + by^2) / bz^2)
end
