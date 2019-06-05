using Oceananigans, Test

@inline incmod1(a, n) = ifelse(a==n, 1, a + 1)
@inline decmod1(a, n) = ifelse(a==1, n, a - 1)
@inline index2rank(I, J, Mx, My) = J*My + I

@inline north_halo(tile) = @views @inbounds tile.data[1-tile.grid.Hx:0, :, :]
@inline south_halo(tile) = @views @inbounds tile.data[tile.grid.Nx+1:tile.grid.Nx+tile.grid.Hx, :, :]
@inline  west_halo(tile) = @views @inbounds tile.data[:, 1-tile.grid.Hy:0, :]
@inline  east_halo(tile) = @views @inbounds tile.data[:, tile.grid.Ny+1:tile.grid.Ny+tile.grid.Hy, :]

@inline north_data(tile) = @views @inbounds tile.data[1:tile.grid.Hx, :, :]
@inline south_data(tile) = @views @inbounds tile.data[tile.grid.Nx-tile.grid.Hx+1:tile.grid.Nx, :, :]
@inline  west_data(tile) = @views @inbounds tile.data[:, 1:tile.grid.Hy, :]
@inline  east_data(tile) = @views @inbounds tile.data[:, tile.grid.Ny-tile.grid.Hy+1:tile.grid.Ny, :]

function fill_halo_regions_tiled!(tiles, Mx, My)
    for J in 0:My-1, I in 0:Mx-1
        rank = index2rank(I, J, Mx, My)

        I⁻, I⁺ = mod(I-1, Mx), mod(I+1, Mx)
        J⁻, J⁺ = mod(J-1, My), mod(J+1, My)

        north_rank = index2rank(I,  J⁻, Mx, My)
        south_rank = index2rank(I,  J⁺, Mx, My)
        east_rank  = index2rank(I⁺, J,  Mx, My)
        west_rank  = index2rank(I⁻, J,  Mx, My)

         east_halo(tiles[rank+1]) .=  west_data(tiles[east_rank+1])
         west_halo(tiles[rank+1]) .=  east_data(tiles[west_rank+1])
        north_halo(tiles[rank+1]) .= south_data(tiles[north_rank+1])
        south_halo(tiles[rank+1]) .= north_data(tiles[south_rank+1])
    end
end

FT, arch = Float64, CPU()

Nx, Ny, Nz = 16, 16, 16
Lx, Ly, Lz = 10, 10, 10
N, L = (Nx, Ny, Nz), (Lx, Ly, Lz)

grid = RegularCartesianGrid(N, L)

# MPI ranks along each dimension
Mx, My = 2, 2

R = rand(Nx, Ny, Nz)

tiles = []
for I in 0:Mx-1, J in 0:My-1
    Nx′, Ny′, Nz′ = Int(Nx/Mx), Int(Ny/My), Nz
    Lx′, Ly′, Lz′ = Lx/Mx, Ly/My, Lz
    tile_grid = RegularCartesianGrid((Nx′, Ny′, Nz′), (Lx′, Ly′, Lz′))

    tile = CellField(FT, arch, tile_grid)

    i1, i2 = I*Nx′+1, (I+1)*Nx′
    j1, j2 = J*Ny′+1, (J+1)*Ny′
    data(tile) .= R[i1:i2, j1:j2, :]

    push!(tiles, tile)
end

fill_halo_regions_tiled!(tiles, Mx, My)
fill_halo_regions_tiled!(tiles, Mx, My)

@test all(tiles[1].data[1:end,     1:end, :] .== R[1:9,   1:9,   :])
@test all(tiles[2].data[1:end,   0:end-1, :] .== R[1:9,   8:end, :])
@test all(tiles[3].data[0:end-1,   1:end, :] .== R[8:end, 1:9,   :])
@test all(tiles[4].data[0:end-1, 0:end-1, :] .== R[8:end, 8:end, :])
