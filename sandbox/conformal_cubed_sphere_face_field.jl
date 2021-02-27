using Oceananigans.Grids: halo_size
using Oceananigans.Fields: AbstractField, validate_field_data

include("conformal_cubed_sphere_grid.jl")

struct ConformalCubedSphereFaceField{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
    boundary_conditions :: B

    function ConformalCubedSphereFaceField{X, Y, Z}(data, grid, bcs) where {X, Y, Z}
        validate_field_data(X, Y, Z, data, grid)
        return new{X, Y, Z, typeof(data), typeof(grid), typeof(bcs)}(data, grid, bcs)
    end
end

function ConformalCubedSphereFaceField(FT::DataType, arch, grid::ConformalCubedSphereFaceGrid, location,
                                        bcs = TracerBoundaryConditions(grid),
                                       data = new_data(FT, arch, grid, location))

    LX, LY, LZ = location
    return ConformalCubedSphereFaceField{LX, LY, LZ}(data, grid, bcs)
end

function show(io::IO, field::ConformalCubedSphereFaceField{LX, LY, LZ}) where {LX, LY, LZ}
    Nx, Ny, Nz = size(field.data) .- 2 .* halo_size(field.grid)
    Hx, Hy, Hz = halo_size(field.grid)
    print(io, "ConformalCubedSphereFaceField{$LX, $LY, $LZ} with size ($Nx, $Ny, $Nz) + 2 × ($Hx, $Hy, $Hz)\n")
end
