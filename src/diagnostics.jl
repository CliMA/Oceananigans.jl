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

mutable struct Nusselt_wT <: Diagnostic
    diagnostic_frequency::Int
    Nu::Array{AbstractFloat,1}
    Nu_inst::Array{AbstractFloat,1}
    wT_cumulative_running_avg::AbstractFloat
end

function run_diagnostic(model::Model, diag::Nusselt_wT)
    α = 207e-6  # Volumetric expansion coefficient [K⁻¹] of water at 20°C.
    g = model.constants.g

    w, T = model.velocities.w.data, model.tracers.T.data
    V = model.grid.Lx * model.grid.Ly * model.grid.Lz
    wT_avg = sum(w .* T) / V

    n = length(diag.Nu)  # Number of "samples" so far.
    diag.wT_cumulative_running_avg = (wT_avg + n*model.clock.Δt*diag.wT_cumulative_running_avg) / ((n+1)*model.clock.Δt)

    Lz, κ, ΔT = model.grid.Lz, model.configuration.κh, 1
    Nu_wT = 1 + (Lz^2 / (κ*α*g*ΔT^2)) * diag.wT_cumulative_running_avg

    push!(diag.Nu, Nu_wT)

    Nu_wT_inst = 1 + (Lz^2 / (κ*α*g*ΔT^2)) * wT_avg
    push!(diag.Nu_inst, Nu_wT_inst)
end

mutable struct Nusselt_Chi <: Diagnostic
    diagnostic_frequency::Int
    Nu::Array{AbstractFloat,1}
    Nu_inst::Array{AbstractFloat,1}
    ∇T²_cumulative_running_avg::AbstractFloat
end

function norm_gradient_squared!(g::RegularCartesianGrid, f::CellField, ∇f²::CellField, stmp::StepperTemporaryFields)
    dfdx, dfdy, dfdz = otmp.fFX, otmp.fFY, otmp.fFZ

    δx!(g, f, dfdx)
    δy!(g, f, dfdy)
    δz!(g, f, dfdz)

    @. ∇f².data = dfdx.data^2 + dfdy.data^2 + dfdz.data^2
    nothing
end

function run_diagnostic(model::Model, diag::Nusselt_Chi)
    T = model.tracers.T
    ∇T² = model.timestepper.tmp.fC1
    norm_gradient_squared!(model.grid, T, ∇T², model.timestepper.tmp)

    V = model.grid.Lx * model.grid.Ly * model.grid.Lz
    ∇T²_avg = sum(∇T²) / V

    n = length(diag.Nu)  # Number of "samples" so far.
    diag.∇T²_cumulative_running_avg = (∇T²_avg + n*model.clock.Δt*diag.∇T²_cumulative_running_avg) / ((n+1)*model.clock.Δt)

    Lz, ΔT = model.grid.Lz, 1
    Nu_Chi = 1 + (Lz/ΔT)^2 * diag.∇T²_cumulative_running_avg

    push!(diag.Nu, Nu_Chi)

    ∇T²_inst = 1 + (Lz/ΔT)^2 * ∇T²_avg
    push!(diag.Nu_inst, ∇T²_inst)
end
