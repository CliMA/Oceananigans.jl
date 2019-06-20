import GPUifyLoops: @launch, @loop, @unroll, @synchronize

"""
    fill_halo_regions!(arch::Architecture, grid::Grid, bcs::ModelBoundaryConditions, fields...)

Fill in the halo regions for each field in `fields` appropriately based on the
models' boundary conditions specified by `bcs`. For now, the two scenarios are
implementing periodic boundary conditions in the horizontal (for a doubly
periodic domain) and placing walls in the y-direction to impose free-slip
boundary conditions for a reentrant channel model.

Knowledge of `arch` and `grid` is needed to fill in the halo regions.
"""
function fill_halo_regions!(arch::Architecture, grid::Grid, bcs::ModelBoundaryConditions, fields...)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    # Indexing over the offset array but this is commented out as it induces
    # scalars operations on CuArrays right now which are extremely inefficient.
    # Below we directly index the underlying Array or CuArray to avoid the issue of
    # broadcasting over an OffsetArray{CuArray}.

    # @views @inbounds @. f[1-Hx:0,     :, :] = f[Nx-Hx+1:Nx, :, :]
    # @views @inbounds @. f[Nx+1:Nx+Hx, :, :] = f[1:Hx,       :, :]
    # @views @inbounds @. f[:,     1-Hy:0, :] = f[:, Ny-Hy+1:Ny, :]
    # @views @inbounds @. f[:, Ny+1:Ny+Hy, :] = f[:,       1:Hy, :]

    if bcs.u.y.left == BoundaryCondition(Periodic, nothing)
        # Doubly periodic domain
        for f in fields
            @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
            @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]
            @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, Ny+1:Ny+Hy, :]
            @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = f.parent[:, 1+Hy:2Hy,   :]
        end
    elseif bcs.u.y.left == BoundaryCondition(FreeSlip, nothing)
        # Reentrant channel model
        for f in fields
            @views @inbounds @. f.parent[1:Hx,           :, :] = f.parent[Nx+1:Nx+Hx, :, :]
            @views @inbounds @. f.parent[Nx+Hx+1:Nx+2Hx, :, :] = f.parent[1+Hx:2Hx,   :, :]
            @views @inbounds @. f.parent[:, 1:Hy,           :] = f.parent[:, 1+Hy:2Hy,   :]
            @views @inbounds @. f.parent[:, Ny+Hy+1:Ny+2Hy, :] = 0
        end
    end
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
