using PyPlot, Glob, Statistics

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFixedSlip.u
v = ConvergenceTests.ForcedFlowFixedSlip.v

filenames = [
             "data/forced_fixed_slip_xy_Nx16_Δt6.0e-06.jld2",
             "data/forced_fixed_slip_xy_Nx32_Δt6.0e-06.jld2",
             "data/forced_fixed_slip_xy_Nx64_Δt6.0e-06.jld2",
             "data/forced_fixed_slip_xy_Nx128_Δt6.0e-06.jld2",
            ]

close("all")
fig, axs = subplots(nrows=2, ncols=2, figsize=(26, 12))

for filename in filenames
    u_sim, u_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> u(x, y, t), filename; name=:u)
    v_sim, v_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> v(x, y, t), filename; name=:v)

    grid = RegularCartesianGrid(filename)

    ulim = maximum(abs, u_ana)
    vlim = maximum(abs, v_ana)

    i = round(Int,     grid.Nx / 4)
    j = round(Int, 3 * grid.Nx / 4)

    u_err = @. abs(u_sim - u_ana)
    v_err = @. abs(v_sim - v_ana)

    @show mean(u_err[:, j, 1])
    @show mean(v_err[:, j, 1])

    sca(axs[1, 1])
    plot(grid.xF[2:end-1], u_err[:, j, 1])
    xlabel("x")
    ylabel(L"u")

    sca(axs[1, 2])
    plot(grid.xC[2:end-1], v_err[:, j, 1])
    xlabel("x")
    ylabel(L"v")

    sca(axs[2, 1])
    plot(grid.yC[2:end-1], u_err[i, :, 1])
    xlabel("y")
    ylabel(L"u")

    sca(axs[2, 2])
    plot(grid.yF[2:end-1], v_err[i, :, 1])
    xlabel("y")
    ylabel(L"v")
end
