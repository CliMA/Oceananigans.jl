macro loop_xyz(i, j, k, grid, expr)
    return esc(
        quote
            @loop for $k in (1:$grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
                @loop for $j in (1:$grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                    @loop for $i in (1:$grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                        $expr
                    end
                end
            end
        end)
end

macro loop_xy(i, j, grid, expr)
    return esc(
        quote
            @loop for $j in (1:$grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for $i in (1:$grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    $expr
                end
            end
        end)
end

macro loop_xz(i, k, grid, expr)
    return esc(
        quote
            @loop for $k in (1:$grid.Nz; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for $i in (1:$grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    $expr
                end
            end
        end)
end
