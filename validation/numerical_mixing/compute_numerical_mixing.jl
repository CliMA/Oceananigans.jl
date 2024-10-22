using Oceananigans
using Oceananigans.Advection: RotatedAdvection

include("compute_rpe.jl")
include("baroclinic_adjustment.jl")

using Oceananigans.TurbulenceClosures
using Oceananigans.Operators: Δx, Δy, Δxᶜᶜᶜ, Δyᶜᶜᶜ
using Oceananigans.Units

@inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
@inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

momentum_advection = WENOVectorInvariant(; vorticity_order = 9)
horizontal_closure = nothing # HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters = 5days)

w3 = WENO(; order = 3)
w5 = WENO(; order = 5)
w7 = WENO(; order = 7)

tracer_advections  = [
    w3,
    w5,
    w7,
    RotatedAdvection(w3),
    RotatedAdvection(w5),
    RotatedAdvection(w7)
]

filenames = [
    "baroclinic_adjustment_weno3",
    "baroclinic_adjustment_weno5",
    "baroclinic_adjustment_weno7",
    "baroclinic_adjustment_rotated_weno3",
    "baroclinic_adjustment_rotated_weno5",
    "baroclinic_adjustment_rotated_weno7"
]

for i in [1, 2, 3, 4, 5, 6]
    sim = baroclinic_adjustment_simulation(1/8, filenames[i]; 
                                           arch = GPU(), 
                                           momentum_advection,
                                           horizontal_closure,
                                           tracer_advection = tracer_advections[i])
    run!(sim)
end