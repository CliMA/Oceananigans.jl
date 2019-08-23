using Statistics: mean
using Printf

@hascuda using CUDAdrv, CUDAnative

function time_to_run(clock::Clock, diag::Diagnostic)
    if :interval in propertynames(diag) && diag.interval != nothing
        if clock.time >= diag.previous + diag.interval
            diag.previous = clock.time - rem(clock.time, diag.interval)
            return true
        else
            return false
        end
    elseif :frequency in propertynames(diag) && diag.frequency != nothing
        return clock.iteration % diag.frequency == 0
    else
        error("Diagnostic $(typeof(diag)) must have a frequency or interval specified!")
    end
end

function validate_interval(frequency, interval)
    if isnothing(frequency) && isnothing(interval)
        error("Must choose either a frequency (number of iterations) or a time interval!")
    elseif isnothing(interval)
        if isinteger(frequency)
            return Int(frequency), interval
        else
            error("Frequency $frequency must be an integer!")
        end
    elseif isnothing(frequency)
        if isa(interval, Number)
            return frequency, Float64(interval)
        else
            error("Interval must be convertable to a float!")
        end
    else
        error("Cannot choose both frequency and interval!")
    end
end

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

@inline _prod(data, ::Nothing, i, j, k) = @inbounds data[i, j, k]
@inline _prod(data1,    data2, i, j, k) = @inbounds data1[i, j, k] * data2[i, j, k]

# Work-inefficient inclusive scan that reduces along the xy dimensions.
# Useful for computing mean(; dim=[1, 2]) lol.
# Modified from: https://github.com/JuliaGPU/CUDAnative.jl/blob/master/examples/scan.jl
@hascuda function gpu_accumulate_xy!(Rxy::CuDeviceArray{T}, Rx, data1, data2, op::Function) where T
    lvl, lvls = blockIdx().y,  gridDim().y
    col, cols = blockIdx().x,  gridDim().x
    row, rows = threadIdx().x, blockDim().x

    if lvl <= lvls && col <= cols && row <= rows
        shmem = @cuDynamicSharedMem(T, 2*rows)
        shmem[row] = _prod(data1, data2, row, col, lvl)
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
            Rxy[1, 1, lvl] = real(sum)
        end
    end

    return
end

####
#### Horizontally averaged vertical profiles
####

mutable struct HorizontalAverage{P, F, I, T} <: Diagnostic
      profile :: P
        field :: F
    frequency :: I
     interval :: T
     previous :: Float64
end

function HorizontalAverage(model, field; frequency=nothing, interval=nothing)
    freq, int = validate_interval(frequency, interval)
    profile = zeros(model.arch, model.grid, 1, 1, model.grid.Nz)
    HorizontalAverage(profile, field, freq, int, 0.0)
end

function run_diagnostic(model::Model, P::HorizontalAverage{<:Array})
    P.profile .= mean(data(P.field), dims=[1, 2])
end

@hascuda function run_diagnostic(model::Model, P::HorizontalAverage{<:CuArray})
    Nx, Ny, Nz = P.field.grid.Nx, P.field.grid.Ny, P.field.grid.Nz
    sz = 2Nx * sizeof(eltype(P.profile))
    @cuda threads=Nx blocks=(Ny, Nz) shmem=sz gpu_accumulate_xy!(P.profile, model.poisson_solver.storage, P.field.data, nothing, +)
    P.profile /= (Nx*Ny)  # Normalize to get the mean from the sum.
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
