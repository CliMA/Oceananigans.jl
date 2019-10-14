include("stratified_couette_flow.jl")

simulate_stratified_couette_flow(Nxy=128, Nz=128, Ri=0)
simulate_stratified_couette_flow(Nxy=128, Nz=128, Ri=0.01)
simulate_stratified_couette_flow(Nxy=128, Nz=128, Ri=0.04)

