 using GPUifyLoops: @loop

####
#### Useful kernels for doing diagnostics
####

function velocity_div!(grid, u, v, w, div)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
            end
        end
    end
end
