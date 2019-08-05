using Statistics: mean, std
using Printf

struct CFLChecker <: Diagnostic end

struct FieldSummary <: Diagnostic
    diagnostic_frequency::Int
    fields::Array{Field,1}
    field_names::Array{AbstractString,1}
end

function run_diagnostic(model::Model, fs::FieldSummary)
    for (field, field_name) in zip(fs.fields, fs.field_names)
        padded_name = lpad(field_name, 4)
        field_min = minimum(field.data)
        field_max = maximum(field.data)
        field_mean = mean(field.data)
        field_abs_mean = mean(abs.(field.data))
        field_std = std(field.data)
        @printf("%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                padded_name, field_min, field_max, field_mean, field_abs_mean, field_std)
    end
end

struct NaNChecker <: Diagnostic
    diagnostic_frequency::Int
    fields::Array{Field,1}
    field_names::Array{AbstractString,1}
end

function run_diagnostic(model::Model, nc::NaNChecker)
    for (field, field_name) in zip(nc.fields, nc.field_names)
        if any(isnan, field.data.parent)  # This is also fast on CuArrays.
            t, i = model.clock.time, model.clock.iteration
            println("time = $t, iteration = $i")
            println("NaN found in $field_name. Aborting simulation.")
            exit(1)
        end
    end
end

struct VelocityDivergenceChecker <: Diagnostic
    diagnostic_frequency::Int
    warn_threshold::Float64
    abort_threshold::Float64
end

function velocity_div!(grid::RegularCartesianGrid, u, v, w, div)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds div[i, j, k] = div_f2c(grid, u, v, w, i, j, k)
            end
        end
    end
end

function run_diagnostic(model::Model, diag::VelocityDivergenceChecker)
    u, v, w = model.velocities.u.data, model.velocities.v.data, model.velocities.w.data
    div = model.stepper_tmp.fC1

    velocity_div!(model.grid, u, v, w, div)
    min_div, mean_div, max_div = minimum(div), mean(div), maximum(div)

    if max(abs(min_div), abs(max_div)) >= diag.warn_threshold
        t, i = model.clock.time, model.clock.iteration
        println("time = $t, iteration = $i")
        println("WARNING: Velocity divergence is high! min=$min_div, mean=$mean_div, max=$max_div")
    end

    if max(abs(min_div), abs(max_div)) >= diag.abort_threshold
        t, i = model.clock.time, model.clock.iteration
        println("time = $t, iteration = $i")
        println("Velocity divergence is too high! min=$min_div, mean=$mean_div, max=$max_div. Aborting simulation.")
        exit(1)
    end
end

time_to_run(clock, diag) = (clock.iteration % diag.diagnostic_frequency) == 0
