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
    isnothing(frequency) && isnothing(interval) && @error "Must specify a frequency or interval!"
    return
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

####
#### Horizontally averaged vertical profiles
####

mutable struct HorizontalAverage{P, F, I, T} <: Diagnostic
      profile :: P
       fields :: F
    frequency :: I
     interval :: T
     previous :: Float64
end

tupleit(t::Tuple) = t
tupleit(a::AbstractArray) = Tuple(a)
tupleit(nt) = tuple(nt)

function HorizontalAverage(model, fields; frequency=nothing, interval=nothing)
    fields = tupleit(fields)
    fields = Tuple([field.data.parent for field in fields])
    validate_interval(frequency, interval)
    profile = zeros(model.arch, model.grid, 1, 1, model.grid.Tz)
    HorizontalAverage(profile, fields, frequency, interval, 0.0)
end

function run_diagnostic(model::Model, P::HorizontalAverage)
    zero_halo_regions!(P.fields, model.grid)
    if length(P.fields) == 1
        sum!(P.profile, P.fields[1])
    else
        tmp = model.pressures.pNHS.data.parent
        @. tmp = *(P.fields...)
        sum!(P.profile, tmp)
    end

    Nx, Ny = model.grid.Nx, model.grid.Ny
    P.profile /= (Nx*Ny)  # Normalize to get the mean from the sum.
end

function (p::HorizontalAverage)(model)
    run_diagnostic(model, p)
    return p.profile
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
