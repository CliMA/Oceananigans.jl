# # "Cosine advection-diffusion" Spatial resolution convergence test

using PyPlot

# Define a few utilities for running tests and unpacking and plotting results

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

run_test = ConvergenceTests.OneDimensionalCosineAdvectionDiffusion.run_test

""" Run advection-diffusion test for all Nx in resolutions. """
function run_convergence_test(κ, U, resolutions...)

    # Determine save time-step
             Lx = 2π
      stop_time = 0.01
              h = Lx / maximum(resolutions)
    proposal_Δt = 1e-3 * min(h / U, h^2 / κ)

    # Adjust time-step
    stop_iteration = round(Int, stop_time / proposal_Δt)
                Δt = stop_time / stop_iteration

    # Run the tests
    results = [run_test(Nx=Nx, Δt=Δt, stop_iteration=stop_iteration, U=U, κ=κ) for Nx in resolutions]

    return results
end

""" Unpack a vector of results associated with a convergence test. """
function unpack_solutions(results)
    c_ana = map(r -> r.cx.analytical, results)
    c_sim = map(r -> r.cx.simulation, results)
    return c_ana, c_sim
end

function unpack_errors(results)
     u_L₁ = map(r -> r.u.L₁, results)
     v_L₁ = map(r -> r.v.L₁, results)
    cx_L₁ = map(r -> r.cx.L₁, results)
    cy_L₁ = map(r -> r.cy.L₁, results)

     u_L∞ = map(r ->  r.u.L∞, results)
     v_L∞ = map(r ->  r.v.L∞, results)
    cx_L∞ = map(r -> r.cx.L∞, results)
    cy_L∞ = map(r -> r.cy.L∞, results)

    return (u_L₁, v_L₁, cx_L₁, cy_L₁,
            u_L∞, v_L∞, cx_L∞, cy_L∞)
end

""" Unpack a vector of grids associated with a convergence test. """
unpack_grids(results) = map(r -> r.grid, results)

#####
##### Run test
#####

Nx = 2 .^ (3:7) # N = 64 through N = 256
diffusion_results = run_convergence_test(1e-1, 0, Nx...)
#advection_results = run_convergence_test(1e-6, 3, Nx...)
#advection_diffusion_results = run_convergence_test(1e-2, 1, Nx...)

#####
##### Plot solution and error profile
#####

all_results = (diffusion_results,) #advection_results, advection_diffusion_results)
names = ("diffusion only",  "advection only",  "advection-diffusion")
linestyles = ("-", "--", ":")
specialcolors = ("xkcd:black", "xkcd:indigo", "xkcd:wine red")

close("all")
fig, axs = subplots(nrows=2, figsize=(12, 6), sharex=true)

for j = 1:1 #length(all_results)

    results = all_results[j]
    name = names[j]
    linestyle = linestyles[j]

    c_ana, c_sim = unpack_solutions(results)
    grids = unpack_grids(results)

    sca(axs[1])

    plot(grids[end].xC, c_ana[end], "-", color=specialcolors[j], alpha=0.2, linewidth=3, label="$name analytical")

    for i in 1:length(c_sim)
        plot(grids[i].xC, c_sim[i], linestyle, color=defaultcolors[i], alpha=0.8, linewidth=1, 
             label="$name simulated, \$ N_x \$ = $(grids[i].Nx)")
    end

    sca(axs[2])

    for i in 1:length(c_sim)
        plot(grids[i].xC, abs.(c_sim[i] .- c_ana[i]), linestyle, color=defaultcolors[i], alpha=0.8, 
             label="$name, \$ N_x \$ = $(grids[i].Nx)")
    end

end

sca(axs[1])
ylabel(L"c")
removespines("top", "right", "bottom")
axs[1].tick_params(bottom=false, labelbottom=false)
legend(loc="upper left", prop=Dict(:size=>7), bbox_to_anchor=(0.01, 0.3, 1.5, 1.0))

sca(axs[2])
axs[2].set_yscale("log")
removespines("top", "right")
xlabel(L"x")
ylabel("Absolute error \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} | \$")
legend(loc="upper left", prop=Dict(:size=>10), bbox_to_anchor=(0.01, 0.3, 1.5, 1.0))

savefig("figs/cosine_advection_diffusion_solutions.png", dpi=480)

#####
##### Plot error convergence
#####

fig, axs = subplots()

for j = 1:length(all_results)
    results = all_results[j]
    name = names[j]
    u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(results)

    @show u_L₁, u_L∞

    common_kwargs = (linestyle="None", color=defaultcolors[j], mfc="None", alpha=0.8)
    loglog(Nx,  u_L₁; marker="o", label="\$L_1\$-norm, \$u\$ $name", common_kwargs...)
    loglog(Nx,  v_L₁; marker="2", label="\$L_1\$-norm, \$v\$ $name", common_kwargs...)
    loglog(Nx, cx_L₁; marker="*", label="\$L_1\$-norm, \$x\$ tracer $name", common_kwargs...)
    loglog(Nx, cy_L₁; marker="+", label="\$L_1\$-norm, \$y\$ tracer $name", common_kwargs...)

    loglog(Nx,  u_L∞; marker="1", label="\$L_\\infty\$-norm, \$u\$ $name", common_kwargs...)
    loglog(Nx,  v_L∞; marker="_", label="\$L_\\infty\$-norm, \$v\$ $name", common_kwargs...)
    loglog(Nx, cx_L∞; marker="^", label="\$L_\\infty\$-norm, \$x\$ tracer $name", common_kwargs...)
    loglog(Nx, cy_L∞; marker="s", label="\$L_\\infty\$-norm, \$y\$ tracer $name", common_kwargs...)
end

# Guide line to confirm second-order scaling
u_L₁, v_L₁, cx_L₁, cy_L₁, u_L∞, v_L∞, cx_L∞, cy_L∞  = unpack_errors(all_results[1])
loglog(Nx, cx_L₁[1] .* (Nx[1] ./ Nx).^2, "k-", alpha=0.8, label=L"\sim N_x^{-2}")

axs.grid(which="both", linewidth=1)
xlabel(L"N_x")
ylabel("\$L\$-norms of \$ | c_\\mathrm{sim} - c_\\mathrm{analytical} |\$")
removespines("top", "right")
legend(loc="upper right", prop=Dict(:size=>6))
xticks(Nx, ["\$ 2^{$(round(Int, log2(n)))} \$" for n in Nx])

savefig("figs/cosine_advection_diffusion_error_convergence.png", dpi=480)
