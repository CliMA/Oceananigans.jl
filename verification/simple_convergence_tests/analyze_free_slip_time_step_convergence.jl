using PyPlot, Glob

include("ForcedFlow/ForcedFlow.jl")


free_slip_simulation_files = [
                              joinpath("data", "forced_free_slip_Nx32_Nz32_CFL1e-04.jld2"),
                              joinpath("data", "forced_free_slip_Nx32_Nz32_CFL5e-04.jld2"),
                              joinpath("data", "forced_free_slip_Nx32_Nz32_CFL1e-03.jld2"),
                              joinpath("data", "forced_free_slip_Nx32_Nz32_CFL2e-03.jld2")
                             ]

Nx, Ny, Nz = size(ForcedFlow.RegularCartesianGrid(free_slip_simulation_files[1]))

CFL = [1e-4, 5e-4, 1e-3, 2e-3]
Δt = [ c * 1/Nz for c in CFL ]

errors = ForcedFlow.compute_errors(ForcedFlow.FreeSlip.u, 
                                   free_slip_simulation_files...)

L₁ = map(err -> err.L₁, errors)
L∞ = map(err -> err.L∞, errors)

fig, ax = subplots()

loglog(Δt, L₁, linestyle="None", marker="o", label="error, \$L_1\$-norm")
loglog(Δt, L∞, linestyle="None", marker="^", label="error, \$L_\\infty\$-norm")

#loglog(Δt, 200 * Nx.^(-2), "k--", linewidth=1, alpha=0.6, label="\$ 200 N_x^{-2} \$")
#loglog(Δt, 5 * Nx.^(-1),   "k-", linewidth=1, alpha=0.6, label="\$ 5 N_x^{-1} \$")

xlabel(L"\Delta t")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")
title("Time step convergence for forced free slip")

legend()
