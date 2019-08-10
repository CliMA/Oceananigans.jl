using Statistics: mean
using Printf

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

####
#### NaN checker
####

struct NaNChecker <: Diagnostic
    frequency  :: Int
       fields  :: Array{Field,1}
    field_names:: Array{AbstractString,1}
end

function run_diagnostic(model::Model, nc::NaNChecker)
    for (field, field_name) in zip(nc.fields, nc.field_names)
        if any(isnan, field.data.parent)  # This is also fast on CuArrays.
            t, i = model.clock.time, model.clock.iteration
            error("time = $t, iteration = $i: NaN found in $field_name. Aborting simulation.")
        end
    end
end

