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

p̆(y) = - 1/4 * y^4 + 1/3 * y^3 + 143/144 * y^2 - 35/18 * y + 1/2 - 289/72 * cosh(y)/sinh(1)
p̃(y) =   1/4 * y^4 - 1/3 * y^3 - 575/144 * y^2 + 71/18 * y + 2   - 721/72 * cosh(y)/sinh(1) + 4 * cosh(y-1)/sinh(1)
p̂(y) = - 10/4 * y^3 - 15/4 * y + 90/16 * cosh(2y)/sinh(2)

#p(x, y, t) = p̂(y) * cos(2*(x - ξ(t))) + p̆(y) * ξ′(t) * cos(x - ξ(t)) + p̃(y) * sin(x - ξ(t))
p(x, y, t) = p̆(y) * ξ′(t) * cos(x - ξ(t)) + p̃(y) * sin(x - ξ(t))

#filename = "data/forced_fixed_slip_xy_Nx16_Δt6.0e-06.jld2"
#filename = "data/forced_fixed_slip_xy_Nx32_Δt6.0e-06.jld2"
#filename = "data/forced_fixed_slip_xy_Nx64_Δt6.0e-06.jld2"
filename = "data/forced_fixed_slip_xy_Nx128_Δt6.0e-06.jld2"

RegularCartesianGrid = ConvergenceTests.RegularCartesianGrid 
grid = RegularCartesianGrid(filename)

u_sim, u_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> u(x, y, t), filename; name=:u)
v_sim, v_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> v(x, y, t), filename; name=:v)
p_sim, p_ana = ConvergenceTests.extract_two_solutions((x, y, z, t) -> p(x, y, t), filename; name=:p)

u_err = @. abs(u_sim - u_ana)
v_err = @. abs(v_sim - v_ana)
p_err = @. abs(p_sim - p_ana)

lim = 0.01

close("all")
fig, axs = subplots(ncols=2, nrows=1)

sca(axs[1])
imshow(u_err[:,  :, 1]' / maximum(abs, u_ana), vmin=0, vmax=lim, cmap="YlGnBu_r")
title("\$ u \$ error")

sca(axs[2])
imshow(v_err[:,  :, 1]' / maximum(abs, v_ana), vmin=0, vmax=lim, cmap="YlGnBu_r")
title("\$ v \$ error")

close("all")
fig, axs = subplots(ncols=3, nrows=1)

plim = maximum(abs, p_ana)

sca(axs[1])
imshow(p_sim[:,  :, 1]', vmin=-plim, vmax=plim, cmap="RdBu_r")

sca(axs[2])
imshow(p_ana[:,  :, 1]', vmin=-plim, vmax=plim, cmap="RdBu_r")

sca(axs[3])
imshow(p_err[:,  :, 1]' / maximum(abs, p_ana), vmin=0, vmax=1.0, cmap="YlGnBu_r")

title("\$ p \$ error")

#nx = size(u_sim, 1) - 2

#=
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

=#
