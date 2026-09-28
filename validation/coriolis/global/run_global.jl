# Spin-up from rest of the global setup with one Coriolis scheme, with daily kinetic energy and enstrophy budgets.
# Usage: julia --project run_global.jl <FourPoint | CD | ShearSigned | FixedSE | FixedNW> [years = 5] [pickup]

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "..", "momentum_budgets.jl"))

using Oceananigans.Grids: λnode
using Oceananigans.Operators: Δzᶠᶜᶜ
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization

@inline function wind_u(i, j, k, grid, clock, p)
    λu = λnode(i, j, k, grid, Face(), Center(), Center())
    φu = φnode(i, j, k, grid, Face(), Center(), Center())
    return ifelse(k == grid.Nz, - zonal_momentum_flux(λu, φu, clock.time, p) / Δzᶠᶜᶜ(i, j, k, grid), zero(grid))
end
@inline wind_v(i, j, k, grid, clock, p) = zero(grid)

function fixed_orientation(χ)
    scheme = ShearSignedCoriolis(grid; adjustment_time=Inf)
    set!(scheme.chirality, χ)
    fill_halo_regions!(scheme.chirality)
    return scheme
end

schemes = Dict("FourPoint"   => () -> EnergyConserving(),
               "CD"          => () -> CDScheme(grid),
               "ShearSigned" => () -> ShearSignedCoriolis(grid),
               "FixedSE"     => () -> fixed_orientation(1),
               "FixedNW"     => () -> fixed_orientation(-1))

name = ARGS[1]
years = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 5
pickup = "pickup" in ARGS
case = "$(name)_$(FT)"

simulation = global_simulation(case; coriolis=HydrostaticSphericalCoriolis(scheme=schemes[name]()), stop_time=years * 365days, pickup)

open(joinpath(OUTPUT, "coordinates.txt"), "w") do io
    for j in 1:Ny, i in 1:Nx
        println(io, λnode(i, j, 1, grid, Center(), Center(), Center()), " ", φnode(i, j, 1, grid, Center(), Center(), Center()))
    end
end

history_file = joinpath(OUTPUT, "$(case)_budget.jld2")
implicit_closure = ConvectiveAdjustmentVerticalDiffusivity(ExplicitTimeDiscretization(); convective_κz=1, convective_νz=1)
budget = MomentumBudget(simulation.model; implicit_closure, surface_flux=(wind_u, wind_v, simulation.model.clock, (; τ₀, ρ₀)))

function save_history(history)
    jldsave(history_file; time=history.time, kinetic_energy=history.kinetic_energy, enstrophy=history.enstrophy,
            energy_rates=history.energy_rates, enstrophy_rates=history.enstrophy_rates)
end

# After a pickup, the saved history is reloaded and truncated at the time of the checkpoint
function restore_history!(history, time)
    jldopen(history_file) do file
        kept = file["time"] .< time
        append!(history.time, file["time"][kept])
        append!(history.kinetic_energy, file["kinetic_energy"][kept])
        append!(history.enstrophy, file["enstrophy"][kept])
        for (key, rates) in file["energy_rates"]
            append!(history.energy_rates[key], rates[kept])
        end
        for (key, rates) in file["enstrophy_rates"]
            append!(history.enstrophy_rates[key], rates[kept])
        end
    end
    return nothing
end

function record(sim)
    pickup && isempty(budget.history.time) && isfile(history_file) && restore_history!(budget.history, time(sim))
    record_budget!(budget)
    round(Int, time(sim) / day) % 30 == 0 && save_history(budget.history)
    return nothing
end

add_callback!(simulation, record, TimeInterval(1day))
run!(simulation; pickup)
save_history(budget.history)
touch(joinpath(OUTPUT, "$case.done"))
