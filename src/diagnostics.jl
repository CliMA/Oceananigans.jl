using Statistics: mean, std
using Printf

struct CFLChecker <: Diagnostic end
struct VelocityDivergence <: Diagnostic end

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
        if any(isnan, field.data)  # This is also fast on CuArrays.
            t, i = model.clock.time, model.clock.iteration
            println("time = $t, iteration = $i")
            println("NaN found in $field_name. Aborting simulation.")
            exit(1)
        end
    end
end
