module Grids

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: AbstractGrid, StaticVerticalDiscretization, MutableVerticalDiscretization
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: GridFittedBottom

import ..OceananigansReactantExt: deconcretize
import Oceananigans.Grids: LatitudeLongitudeGrid, RectilinearGrid
import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

const ReactantGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:ReactantState}

deconcretize(z::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(
        deconcretize(z.cᵃᵃᶠ),
        deconcretize(z.cᵃᵃᶜ),
        deconcretize(z.Δᵃᵃᶠ),
        deconcretize(z.Δᵃᵃᶜ)
    )

# TODO: handle MutableVerticalDiscretization in grid constructors
deconcretize(z::MutableVerticalDiscretization) = z
    
function LatitudeLongitudeGrid(arch::ReactantState, FT::DataType; kw...)
    cpu_grid = LatitudeLongitudeGrid(CPU(), FT; kw...)
    other_names = propertynames(cpu_grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
    return LatitudeLongitudeGrid{TX, TY, TZ}(arch, other_properties...)
end

function RectilinearGrid(arch::ReactantState, FT::DataType; kw...)
    cpu_grid = RectilinearGrid(CPU(), FT; kw...)
    other_names = propertynames(cpu_grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
    return RectilinearGrid{TX, TY, TZ}(arch, other_properties...)
end

function OrthogonalSphericalShellGrid{TX, TY, TZ}(arch::ReactantState,
                                                  Nx, Ny, Nz, Hx, Hy, Hz,
                                                  Lz :: FT,
                                                  λᶜᶜᵃ :: A, λᶠᶜᵃ :: A, λᶜᶠᵃ :: A, λᶠᶠᵃ :: A,
                                                  φᶜᶜᵃ :: A, φᶠᶜᵃ :: A, φᶜᶠᵃ :: A, φᶠᶠᵃ :: A,
                                                  z :: Z,
                                                  Δxᶜᶜᵃ :: A, Δxᶠᶜᵃ :: A, Δxᶜᶠᵃ :: A, Δxᶠᶠᵃ :: A,
                                                  Δyᶜᶜᵃ :: A, Δyᶜᶠᵃ :: A, Δyᶠᶜᵃ :: A, Δyᶠᶠᵃ :: A, 
                                                  Azᶜᶜᵃ :: A, Azᶠᶜᵃ :: A, Azᶜᶠᵃ :: A, Azᶠᶠᵃ :: A,
                                                  radius :: FT,
                                                  conformal_mapping :: C) where {TX, TY, TZ, FT, Z, A, C}

    args = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
            φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
            z,
            Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
            Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
            Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    dargs = Tuple(deconcretize(a) for a in args)
    Arch = typeof(arch)

    return OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, A, C, Arch}(arch, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                                                                       dargs..., radius, conformal_mapping)
end

deconcretize(gfb::GridFittedBottom) = GridFittedBottom(deconcretize(gfb.bottom_height),
                                                       gfb.immersed_condition)

function with_cpu_architecture(::CPU, grid::ReactantGrid)
    other_names = propertynames(grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(grid)
    GridType = typeof(grid).name.wrapper
    return GridType{TX, TY, TZ}(CPU(), other_properties...)
end

function reactant_immersed_boundary_grid(grid, ib; active_cells_map)
    cpu_grid = with_cpu_architecture(CPU(), grid)
    ibg = ImmersedBoundaryGrid(cpu_grid, ib; active_cells_map)
    TX, TY, TZ = Oceananigans.Grids.topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ibg.immersed_boundary,
                                            ibg.interior_active_cells, ibg.active_z_columns)
end

ImmersedBoundaryGrid(grid::ReactantGrid, ib::GridFittedBottom; active_cells_map::Bool=true) =
    reactant_immersed_boundary_grid(grid, ib; active_cells_map)

ImmersedBoundaryGrid(grid::ReactantGrid, ib; active_cells_map::Bool=true) =
    reactant_immersed_boundary_grid(grid, ib; active_cells_map)
    


end # module

