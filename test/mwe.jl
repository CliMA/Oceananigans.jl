using Oceananigans
using Reactant
using KernelAbstractions: @kernel, @index

function simple_tendency!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.Utils.launch!(
        arch,
        grid,
        :xyz,
        _simple_tendency_kernel!,
        model.timestepper.Gⁿ.u,
        grid,
        model.advection.momentum,
        model.velocities)
    return nothing
end

@kernel function _simple_tendency_kernel!(Gu, grid, advection, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = - Oceananigans.Advection.U_dot_∇u(i, j, k, grid, advection, velocities)
end

Nx, Ny, Nz = (10, 10, 10) # number of cells
halo = (7, 7, 7)
longitude = (0, 4)
stretched_longitude = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.3, 2.5, 2.6, 3.5, 4.0]
latitude = (0, 4)
z = (-1, 0)
lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
hydrostatic_model_kw = (; momentum_advection=VectorInvariant(), free_surface=ExplicitFreeSurface())

arch = Oceananigans.Architectures.ReactantState()
grid = LatitudeLongitudeGrid(arch; lat_lon_kw...)
model = HydrostaticFreeSurfaceModel(; grid, hydrostatic_model_kw...)

ui = randn(size(model.velocities.u)...)
vi = randn(size(model.velocities.v)...)
set!(model, u=ui, v=vi)

u, v, w = model.velocities
ui = Array(interior(u))
vi = Array(interior(v))
wi = Array(interior(w))

@show maximum(abs.(ui))
@show maximum(abs.(vi))
@show maximum(abs.(wi))

@jit simple_tendency!(model)

Gu = model.timestepper.Gⁿ.u
Gv = model.timestepper.Gⁿ.v
Gui = Array(interior(Gu))
Gvi = Array(interior(Gv))

@show maximum(abs.(Gui))
@show maximum(abs.(Gvi))

