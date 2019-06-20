import GPUifyLoops: @launch, @loop, @unroll, @synchronize

function fill_halo_regions!(arch::Architecture, grid::Grid, fields...)
"""
    fill_halo_regions!(arch::Architecture, grid::Grid, bcs::ModelBoundaryConditions, fields...)

Fill in the halo regions for each field in `fields` appropriately based on the
models' boundary conditions specified by `bcs`. For now, the two scenarios are
implementing periodic boundary conditions in the horizontal (for a doubly
periodic domain) and placing walls in the y-direction to impose free-slip
boundary conditions for a reentrant channel model.

Knowledge of `arch` and `grid` is needed to fill in the halo regions.
"""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    for f in fields
        # Index over the offset array but this is commented out as it induces
        # scalars operations on CuArrays right now which are extremely inefficient.
        # @views @inbounds @. f[1-Hx:0,     :, :] = f[Nx-Hx+1:Nx, :, :]
        # @views @inbounds @. f[Nx+1:Nx+Hx, :, :] = f[1:Hx,       :, :]
        # @views @inbounds @. f[:,     1-Hy:0, :] = f[:, Ny-Hy+1:Ny, :]
        # @views @inbounds @. f[:, Ny+1:Ny+Hy, :] = f[:,       1:Hy, :]

        # Directly index the underlying Array or CuArray to avoid the issue of
        # broadcasting over an OffsetArray{CuArray}.
        @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
        @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]
        @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
        @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
    end

    # max_threads = 256
    #
    # Ty = min(max_threads, Ny)
    # Tz = min(fld(max_threads, Ty), Nz)
    # By, Bz = cld(Ny, Ty), cld(Nz, Tz)
    # @launch device(arch) threads=(1, Ty, Tz) blocks=(1, By, Nz) fill_halo_regions_x!(grid, fields...)
    #
    # Tx = min(max_threads, Nx)
    # Tz = min(fld(max_threads, Tx), Nz)
    # Bx, Bz = cld(Nx, Tx), cld(Nz, Tz)
    # @launch device(arch) threads=(Tx, 1, Tz) blocks=(Bz, 1, Nz) fill_halo_regions_y!(grid, fields...)
end

"""
    fill_halo_regions_x!(grid::Grid, fields...)

Kernel that fill in the "eastern" and "western" halo regions for each field in
`fields` to impose horizontally periodic boundary conditions.
"""
function fill_halo_regions_x!(grid::Grid, fields...)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz  # Number of grid points.
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz  # Size of halo regions.

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @unroll for f in fields
                @unroll for h in 1:Hx
                    f[1-h,  j, k] = f[Nx-h+1, j, k]
                    f[Nx+h, j, k] = f[h,      j, k]
                end
            end
        end
    end

    @synchronize
end

"""
    fill_halo_regions_y!(grid::Grid, fields...)

Kernel that fill in the "northern" and "southern" halo regions for each field in
`fields` to impose horizontally periodic boundary conditions.
"""
function fill_halo_regions_y!(grid::Grid, fields...)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz  # Number of grid points.
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz  # Size of halo regions.

    @loop for k in (1:Nz; blockIdx().z)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @unroll for f in fields
                @unroll for h in 1:Hy
                    f[i,  1-h, k] = f[i, Ny-h+1, k]
                    f[i, Ny+h, k] = f[i,      h, k]
                end
            end
        end
    end

    @synchronize
end
