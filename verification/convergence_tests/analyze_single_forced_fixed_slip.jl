using PyPlot, Glob

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFixedSlip.u
v = ConvergenceTests.ForcedFlowFixedSlip.v

#filename = "data/forced_fixed_slip_xy_Nx16_Δt6.0e-06.jld2"
#filename = "data/forced_fixed_slip_xy_Nx32_Δt6.0e-06.jld2"
#filename = "data/forced_fixed_slip_xy_Nx64_Δt6.0e-06.jld2"
filename = "data/forced_fixed_slip_xy_Nx128_Δt6.0e-06.jld2"

u_sim, u_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> u(x, y, t), filename; name=:u)
v_sim, v_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> v(x, y, t), filename; name=:v)

ulim = maximum(abs, u_ana)
vlim = maximum(abs, v_ana)

close("all")
fig, axs = subplots(ncols=3, nrows=2)

sca(axs[1, 1])
imshow(u_sim[:,  :, 1]', vmin=-ulim, vmax=ulim, cmap="RdBu_r")
title("\$ u(x, y, t) \$, simulation")

sca(axs[1, 2])
imshow(u_ana[:,  :, 1]', vmin=-ulim, vmax=ulim, cmap="RdBu_r")
title("\$ u(x, y, t) \$ analytical")

sca(axs[1, 3])
imshow((u_sim[:, :, 1] .- u_ana[:,  :, 1])', vmin=-ulim, vmax=ulim, cmap="RdBu_r")
title("Difference")

sca(axs[2, 1])
imshow(v_sim[:,  :, 1]', vmin=-vlim, vmax=vlim, cmap="RdBu_r")
title("\$ v(x, y, t) \$ simulation")

sca(axs[2, 2])
imshow(v_ana[:,  :, 1]', vmin=-vlim, vmax=vlim, cmap="RdBu_r")
title("\$ v(x, y, t) \$ analytical")

sca(axs[2, 3])
imshow((v_sim[:, :, 1] .- v_ana[:,  :, 1])', vmin=-vlim, vmax=vlim, cmap="RdBu_r")

nx = size(u_sim, 1) - 2

fig, axs = subplots(nrows=2, ncols=2, figsize=(26, 12))

i = round(Int,  nx/4)
j = round(Int, 3nx/4)

sca(axs[1, 1])
plot(u_sim[:, j, 1])
plot(u_ana[:, j, 1])
xlabel("x")
ylabel(L"u")

sca(axs[1, 2])
plot(v_sim[:, j, 1])
plot(v_ana[:, j, 1])
xlabel("x")
ylabel(L"v")

sca(axs[2, 1])
plot(u_sim[i, :, 1])
plot(u_ana[i, :, 1])
xlabel("y")
ylabel(L"u")

sca(axs[2, 2])
plot(v_sim[i, :, 1])
plot(v_ana[i, :, 1])
xlabel("y")
ylabel(L"v")


