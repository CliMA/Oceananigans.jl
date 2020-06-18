using PyPlot, Glob, Oceananigans

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFixedSlip.u
v = ConvergenceTests.ForcedFlowFixedSlip.v
ξ = ConvergenceTests.ForcedFlowFixedSlip.ξ
ξ′ = ConvergenceTests.ForcedFlowFixedSlip.ξ′
f = ConvergenceTests.ForcedFlowFixedSlip.f
fₓ = ConvergenceTests.ForcedFlowFixedSlip.fₓ

p1(y) = y^3 * (3y - 4) / 12
p2a(y) = y^4 / 4 - y^3 / 3 - 3y^2 + 2y - 6
p2b(y) = 4 * (1 + exp(1) * (4 + exp(1)) ) * cosh(y) / (exp(1)^2 - 1) - 4 * sinh(y)

p(x, y, t) = p1(y) * ξ′(t) * cos(x - ξ(t)) + (p2a(y) + p2b(y)) * sin(x - ξ(t))

filename = "data/forced_fixed_slip_xy_Nx128_Δt6.0e-06.jld2"

RegularCartesianGrid = ConvergenceTests.RegularCartesianGrid 
grid = RegularCartesianGrid(filename)

u_sim, u_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> u(x, y, t), filename; name=:u)
v_sim, v_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> v(x, y, t), filename; name=:v)
p_sim, p_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> p(x, y, t), filename; name=:p)

elim = 1e-2
ulim = maximum(abs, u_ana)
plim = maximum(abs, p_ana)

u_err = @. abs(u_sim - u_ana) / ulim
v_err = @. abs(v_sim - v_ana) / ulim
p_err = @. abs(p_sim - p_ana) / plim

close("all")
fig, axs = subplots(ncols=3, nrows=2)

sca(axs[1, 1])
imshow(u_sim[:,  :, 1]', vmin=-ulim, vmax=ulim, cmap="RdBu_r")
title("\$ u \$ simulation")

sca(axs[2, 1])
imshow(u_err[:,  :, 1]', vmin=0, vmax=elim, cmap="YlGnBu_r")
title("\$ u \$ error")

sca(axs[1, 2])
imshow(v_sim[:,  :, 1]', vmin=-ulim, vmax=ulim, cmap="RdBu_r")
title("\$ v \$ simulation")

sca(axs[2, 2])
imshow(v_err[:,  :, 1]', vmin=0, vmax=elim, cmap="YlGnBu_r")
title("\$ v \$ error")

sca(axs[1, 3])
imshow(p_sim[:,  :, 1]', vmin=-plim, vmax=plim, cmap="RdBu_r")
title("\$ p \$ simulation")

sca(axs[2, 3])
imshow(p_err[:,  :, 1]', vmin=0, vmax=elim, cmap="YlGnBu_r")
title("\$ p \$ error")
