using Statistics: mean
using Printf

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

mutable struct HorizontalAverage{F, R, P, I, T} <: AbstractDiagnostic
        profile :: P
         fields :: F
      frequency :: I
       interval :: T
       previous :: Float64
    return_type :: R
end


function HorizontalAverage(model, fields; frequency=nothing, interval=nothing,
                           return_type=nothing)
    fields = parenttuple(tupleit(fields))
    validate_interval(frequency, interval)
    profile = zeros(model.architecture, model.grid, 1, 1, model.grid.Tz)
    return HorizontalAverage(profile, fields, frequency, interval, 0.0, return_type)
end

"Normalize a horizontal sum to get the horizontal average."
normalize_horizontal_sum!(hsum, grid) = hsum.profile /= (grid.Nx * grid.Ny)

"""
    run_diagnostic(model, havg)

Compute the horizontal average of `havg.fields` and store the
result in `havg.profile`. If length(fields) > 1, compute the
product of the elements of fields (without taking into account
the possibility that they may have different locations in the
staggered grid) before computing the horizontal average.
"""
function run_diagnostic(model::Model, havg::HorizontalAverage{NTuple{1}})
    zero_halo_regions!(havg.fields[1], model.grid)
    sum!(havg.profile, havg.fields[1])
    normalize_horizontal_sum!(havg, model.grid)
    return
end

function run_diagnostic(model::Model, havg::HorizontalAverage)
    zero_halo_regions!(havg.fields, model.grid)

    # Use pressure as scratch space for the product of fields.
    tmp = model.pressures.pNHS.data.parent
    zero_halo_regions!(tmp, model.grid)

    @. tmp = *(havg.fields...)
    sum!(havg.profile, tmp)
    normalize_horizontal_sum!(havg, model.grid)

    return
end

function (havg::HorizontalAverage{F, Nothing})(model) where F
    run_diagnostic(model, havg)
    return havg.profile
end

function (havg::HorizontalAverage)(model)
    run_diagnostic(model, havg)
    return havg.return_type(havg.profile)
end


####
#### NaN checker
####

struct NaNChecker{D} <: AbstractDiagnostic
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
