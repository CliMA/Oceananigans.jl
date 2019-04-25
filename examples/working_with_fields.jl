using Pkg; Pkg.activate("..")

using Oceananigans, PyPlot

import PyPlot: Plot

zplot(c::CellField, args...; kwargs...) = plot(znodes(c), data(c), args...; kwargs...)

grid = RegularCartesianGrid((1, 1, 128), (1, 1, 1))
T = CellField(grid)

H = grid.Lz

softstep(z, d) = 0.5*(tanh(z/d) + 1)
T0(x, y, z) = 20 - 0.01 * z + 0.001 * softstep(z+H/2, H/10) * rand()
model.tracers.T = T0

zplot(model.tracers.T)
