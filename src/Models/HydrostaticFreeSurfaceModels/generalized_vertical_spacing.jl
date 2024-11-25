using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, halo_size, with_halo
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Operators: ∂t_e₃
using Adapt 
using Printf

import Oceananigans.Architectures: arch_array

#####
##### General implementation
#####

update_grid!(model, grid; parameters = :xy) = nothing

#####
##### Additional terms to be included in the momentum equations (fallbacks)
#####

@inline grid_slope_contribution_x(i, j, k, grid, args...) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, args...) = zero(grid)