using Oceananigans.ImmersedBoundaries: ImmersedBoundaries

#####
##### `RightFaceFolded` topologies have prognostic values at Ny+1 for `Face` fields in y
##### This file extends all kernel parameters to Ny + 1 for `RightFaceFolded` topologies so that
##### the extra row in y is correctly computed
#####
                                    #  FT     TX     TY
const RFTRG = TripolarGridOfSomeKind{<:Any, <:Any, <:RightFaceFolded}

# Kernels are typically launched
# - grid dependent parameters (`surface_kernel_parameters(grid)`, `volume_kernel_parameters(grid)`, `diffusivity_kernel_parameters(grid)`, `buffer_surface_kernel_parameters(grid)`...)
# - symbols (:xyz, :xy, ...)
# - `active_cells_map`s

# kernels launched with
# - `surface_kernel_parameters`,
# - `volume_kernel_parameters`,
# - `diffusivity_kernel_parameters`
# do not need any change because they cover `-H+2:N+H-1` where H is 1-larger in the y-direction for RFTRG

# TODO: fix also the "buffer" kernel parameters functions to allow asynchronous distributed RFTRG
# TODO: fix also the "buffer" active cells map to allow for distributed asynchronous RFTRG

#####
##### Covers the symbols kernel parameters as well as
##### `interior_tendency_kernel_parameters(grid)` and `diffusivity_kernel_parameters(grid)`
##### and the active cells maps which use `worksize` to find all the active indices
#####

Utils.worksize(grid::RFTRG) = grid.Nx, grid.Ny+1, grid.Nz
