module Grids

export constant_with_reactant_state

using Reactant

using Oceananigans
using Oceananigans: Distributed
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, StaticVerticalDiscretization, MutableVerticalDiscretization
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: GridFittedBottom, AbstractImmersedBoundary

import ..OceananigansReactantExt: deconcretize
import Oceananigans.Grids: LatitudeLongitudeGrid, RectilinearGrid, OrthogonalSphericalShellGrid
import Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, materialize_immersed_boundary

const ReactantGrid{FT, TX, TY, TZ} = Union{
    AbstractGrid{FT, TX, TY, TZ, <:ReactantState},
    AbstractGrid{FT, TX, TY, TZ, <:Distributed{<:ReactantState}}
}

const ReactantImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S} = Union{
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:ReactantState},
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:Distributed{<:ReactantState}},
}

const ReactantUnderlyingGrid{FT, TX, TY, TZ, CZ} = Union{
    AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:ReactantState},
    AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:Distributed{<:ReactantState}},
}

deconcretize(z::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(
        deconcretize(z.cᵃᵃᶠ),
        deconcretize(z.cᵃᵃᶜ),
        deconcretize(z.Δᵃᵃᶠ),
        deconcretize(z.Δᵃᵃᶜ)
    )

# TODO: handle MutableVerticalDiscretization in grid constructors
deconcretize(z::MutableVerticalDiscretization) = z

# function LatitudeLongitudeGrid(arch::Union{ReactantState, Distributed{<:ReactantState}}, FT::DataType; kw...)
#     cpu_grid = LatitudeLongitudeGrid(CPU(), FT; kw...)
#     other_names = propertynames(cpu_grid)[2:end] # exclude architecture
#     other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
#     TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
#     return LatitudeLongitudeGrid{TX, TY, TZ}(arch, other_properties...)
# end

# function RectilinearGrid(arch::Union{ReactantState, Distributed{<:ReactantState}}, FT::DataType; kw...)
#     cpu_grid = RectilinearGrid(CPU(), FT; kw...)
#     other_names = propertynames(cpu_grid)[2:end] # exclude architecture
#     other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
#     TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
#     return RectilinearGrid{TX, TY, TZ}(arch, other_properties...)
# end

# function OrthogonalSphericalShellGrid(arch::Union{ReactantState, Distributed{<:ReactantState}}, FT::DataType; kw...)
#     cpu_grid = OrthogonalSphericalShellGrid(CPU(), FT; kw...)
#     other_names = propertynames(cpu_grid)[2:end] # exclude architecture
#     other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
#     TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
#     return OrthogonalSphericalShellGrid{TX, TY, TZ}(arch, other_properties...)
# end

# # This is a kind of OrthogonalSphericalShellGrid
# function RotatedLatitudeLongitudeGrid(arch::Union{ReactantState, Distributed{<:ReactantState}}, FT::DataType; kw...)
#     cpu_grid = RotatedLatitudeLongitudeGrid(CPU(), FT; kw...)
#     other_names = propertynames(cpu_grid)[2:end] # exclude architecture
#     other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
#     TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
#     return OrthogonalSphericalShellGrid{TX, TY, TZ}(arch, other_properties...)
# end

# # This low-level constructor supports the external package OrthogonalSphericalShellGrids.jl.
# function OrthogonalSphericalShellGrid{TX, TY, TZ}(arch::Union{ReactantState, Distributed{<:ReactantState}},
#                                                   Nx, Ny, Nz, Hx, Hy, Hz,
#                                                      Lz :: FT,
#                                                    λᶜᶜᵃ :: CC,  λᶠᶜᵃ :: FC,  λᶜᶠᵃ :: CF,  λᶠᶠᵃ :: FF,
#                                                    φᶜᶜᵃ :: CC,  φᶠᶜᵃ :: FC,  φᶜᶠᵃ :: CF,  φᶠᶠᵃ :: FF, z :: Z,
#                                                   Δxᶜᶜᵃ :: CC, Δxᶠᶜᵃ :: FC, Δxᶜᶠᵃ :: CF, Δxᶠᶠᵃ :: FF,
#                                                   Δyᶜᶜᵃ :: CC, Δyᶠᶜᵃ :: FC, Δyᶜᶠᵃ :: CF, Δyᶠᶠᵃ :: FF, 
#                                                   Azᶜᶜᵃ :: CC, Azᶠᶜᵃ :: FC, Azᶜᶠᵃ :: CF, Azᶠᶠᵃ :: FF,
#                                                  radius :: FT,
#                                                   conformal_mapping :: Map) where {TX, TY, TZ, FT, Z, Map,
#                                                                                    CC, FC, CF, FF}

#     args1 = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
#              φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)

#     args2 = (Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
#              Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
#              Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

#     dargs1 = Tuple(deconcretize(a) for a in args1)
#     dz = deconcretize(z)
#     dargs2 = Tuple(deconcretize(a) for a in args2)

#     Arch = typeof(arch)
#     DCC = typeof(dargs1[1]) # deconcretized
#     DFC = typeof(dargs1[2]) # deconcretized
#     DCF = typeof(dargs1[3]) # deconcretized
#     DFF = typeof(dargs1[4]) # deconcretized
#     DZ = typeof(dz) # deconcretized

#     return OrthogonalSphericalShellGrid{FT, TX, TY, TZ, DZ, Map,
#                                         DCC, DFC, DCF, DFF, Arch}(arch, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
#                                                                   dargs1..., dz, dargs2..., radius, conformal_mapping)
# end

function materialize_immersed_boundary(grid::ReactantGrid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    new_ib = GridFittedBottom(bottom_field)
    return new_ib
end

deconcretize(gfb::GridFittedBottom) = GridFittedBottom(deconcretize(gfb.bottom_height),
                                                       gfb.immersed_condition)

#=
function with_cpu_architecture(::CPU, grid::ReactantGrid)
    other_names = propertynames(grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(grid)
    GridType = typeof(grid).name.wrapper
    return GridType{TX, TY, TZ}(CPU(), other_properties...)
end

function reactant_immersed_boundary_grid(grid, ib; active_cells_map, active_z_columns)
    cpu_grid = with_cpu_architecture(CPU(), grid)
    ibg = ImmersedBoundaryGrid(cpu_grid, ib; active_cells_map, active_z_columns)
    TX, TY, TZ = Oceananigans.Grids.topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ibg.immersed_boundary,
                                            ibg.interior_active_cells, ibg.active_z_columns)
end

function ImmersedBoundaryGrid(grid::ReactantUnderlyingGrid, ib::AbstractImmersedBoundary;
                              active_cells_map::Bool=false,
                              active_z_columns::Bool=active_cells_map)

    return reactant_immersed_boundary_grid(grid, ib; active_cells_map, active_z_columns)
end
=#

const CPUUnderlyingGrid{FT, TX, TY, TZ, CZ} = AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:CPU}
const CPUImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S} =
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:CPU}

function constant_with_reactant_state(cpu_grid::AbstractUnderlyingGrid)
    GridType = typeof(cpu_grid).name.wrapper
    other_names = propertynames(cpu_grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
    return GridType{TX, TY, TZ}(ReactantState(), other_properties...)
end

function constant_with_reactant_state(cpu_ibg::CPUImmersedBoundaryGrid)
    underlying_grid = constant_with_reactant_state(cpu_ibg.underlying_grid)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_ibg)
    return ImmersedBoundaryGrid{TX, TY, TZ}(underlying_grid,
                                            cpu_ibg.immersed_boundary,
                                            cpu_ibg.interior_active_cells,
                                            cpu_ibg.active_z_columns)
end

end # module

