# Experiment (not src): hierarchical ghosts, with the advecting velocity of upwind schemes interpolated by Centered(order=2), as in
# the #6267 branch (its WENO constructor sets advecting_velocity_scheme = Centered(order=2); main uses Centered(order=order-1)).
include(joinpath(@__DIR__, "hierarchical.jl"))

@eval Oceananigans.Advection begin
@inline symmetric_interpolate(i, j, k, grid, dir, ℓ, scheme::AUAS, args...) =
    symmetric_interpolate(i, j, k, grid, dir, ℓ, Centered{1, eltype(grid)}(nothing, scheme.time_discretization), args...)
end
