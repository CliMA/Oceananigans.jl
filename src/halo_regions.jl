import GPUifyLoops: @launch, @loop, @unroll

const PeriodicBC = BoundaryCondition(Periodic, nothing)
const NoFluxBC = BoundaryCondition(Flux, 0)

function fill_halo_regions!(grid, field_tuples...)
    for ft in field_tuples
        field, fbcs, data = ft
        fill_halo_region!(grid, Val(field), fbcs, data)
    end
end

function fill_halo_region!(grid::Grid, ::Val{:u}, bcs, f)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
    @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]

    if bcs.y.left == PeriodicBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    elseif bcs.y.left == NoFluxBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = -f.parent[:, Ny+1:Ny+Hy, :]
    end
end

function fill_halo_region!(grid::Grid, ::Val{:v}, bcs, f)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
    @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]

    if bcs.y.left == PeriodicBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    elseif bcs.y.left == NoFluxBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = 0

        f.parent[:, 1+Hy, :] .= 0  # Enforce v=0 at the wall
    end
end

function fill_halo_region!(grid::Grid, ::Val{:w}, bcs, f)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
    @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]

    if bcs.y.left == PeriodicBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    elseif bcs.y.left == NoFluxBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = -f.parent[:, Ny+1:Ny+Hy, :]
    end
end

function fill_halo_region!(grid::Grid, ::Val{:T}, bcs, f)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
    @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]

    if bcs.y.left == PeriodicBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    elseif bcs.y.left == NoFluxBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, Ny+1:Ny+Hy, :]
    end
end

function fill_halo_region!(grid::Grid, ::Val{:S}, bcs, f)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
    @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]

    if bcs.y.left == PeriodicBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    elseif bcs.y.left == NoFluxBC
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, Ny+1:Ny+Hy, :]
    end
end
