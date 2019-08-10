using Statistics: mean
using Printf

@hascuda using CUDAdrv, CUDAnative

time_to_run(clock, diag) = (clock.iteration % diag.frequency) == 0

####
#### Useful kernels
####

function velocity_div!(grid::RegularCartesianGrid, u, v, w, div)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds div[i, j, k] = div_f2c(grid, u, v, w, i, j, k)
            end
        end
    end
end

# Work-inefficient inclusive 3D scan that reduces along the xy dimensions.
# Modified from: https://github.com/JuliaGPU/CUDAnative.jl/blob/master/examples/scan.jl
@hascuda function gpu_accumulate_xy!(Rxy::CuDeviceArray{T}, Rx::CuDeviceArray{T}, data::CuDeviceArray{T}, op::Function) where {T}
    lvl, lvls = blockIdx().y,  gridDim().y
    col, cols = blockIdx().x,  gridDim().x
    row, rows = threadIdx().x, blockDim().x

    if lvl <= lvls && col <= cols && row <= rows
        shmem = @cuDynamicSharedMem(T, 2*rows)
        shmem[row] = data[row, col, lvl]
        sync_threads()

        # parallel reduction
        pin, pout, offset = 1, 0, 1
        while offset < rows
            pout = 1 - pout
            pin = 1 - pin

            if row > offset
                shmem[pout * rows + row] = op(shmem[pin * rows + row], shmem[pin * rows + row - offset])
            else
                 shmem[pout * rows + row] = shmem[pin * rows + row]
            end

            sync_threads()
            offset *= UInt32(2)
        end

        shmem[pin * rows + row] = shmem[pout * rows + row]
        sync_threads()

        # write results
        if row == rows
            Rx[1, col, lvl] = shmem[row]
        end
        sync_threads()

        if col == cols
            sum = 0
            for j in 1:cols
                sum += Rx[1, j, lvl]
            end
            Rxy[1, 1, lvl] = sum
        end
    end

    return
end

####
#### NaN checker
####

struct NaNChecker{D} <: Diagnostic
    frequency :: Int
       fields :: D
end

function NaNChecker(model; frequency=1000, fields=Dict(:w => model.velocities.w.data.parent))
    NaNChecker(frequency, fields)
end

function run_diagnostic(model::Model, nc::NaNChecker)
    for (name, field) in nc.fields
        if any(isnan, field)
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $name. Aborting simulation.")
        end
    end
end
