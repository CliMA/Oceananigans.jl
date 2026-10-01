using Oceananigans

include("compute_rpe.jl")
include("baroclinic_adjustment.jl")

using Oceananigans.TurbulenceClosures
using Oceananigans.Operators: Δx, Δy, Δxᶜᶜᶜ, Δyᶜᶜᶜ
using Oceananigans.Units

@inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
@inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

momentum_advection = WENOVectorInvariant(; vorticity_order = 9)
horizontal_closure = nothing

sim = baroclinic_adjustment_simulation(1/8, "baroclinic_adjustment_QAB2_15minutes"; 
                                        arch = GPU(), Δt = 12.5minutes, 
                                        momentum_advection,
                                        timestepper = :QuasiAdamsBashforth2,
                                        horizontal_closure,
                                        tracer_advection = WENO(; order = 7))
run!(sim)

# w3 = WENO(; order = 3)
# w5 = WENO(; order = 5)
# w7 = WENO(; order = 7)

# tracer_advections  = [
#     w3,
#     RotatedAdvection(w3),
#     w5,
#     RotatedAdvection(w5),
#     w7,
#     RotatedAdvection(w7)
# ]

# filenames = [
#     "baroclinic_adjustment_linear_weno3",
#     "baroclinic_adjustment_linear_rotated_weno3",
#     "baroclinic_adjustment_linear_weno5",
#     "baroclinic_adjustment_linear_rotated_weno5",
#     "baroclinic_adjustment_linear_weno7",
#     "baroclinic_adjustment_linear_rotated_weno7"
# ]

# for i in [2, 4, 6]
#     sim = baroclinic_adjustment_simulation(1/8, filenames[i]; 
#                                            arch = GPU(), 
#                                            momentum_advection,
#                                            horizontal_closure,
#                                            tracer_advection = tracer_advections[i])
#     run!(sim)
# end
