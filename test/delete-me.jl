include("dependencies_for_runtests.jl")
include("utils_for_runtests.jl")
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization, z_viscosity
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom

function center_clustered(N, L, x₀)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + 3 - Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ x₀
    return z_faces
end

function boundary_clustered(N, L, x₀)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ x₀ 
    return z_faces
end

timestepper = :QuasiAdamsBashforth2
time_discretization = ExplicitTimeDiscretization()

Nz, Lz = 128, π/2
grid = RectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz))

# grid = RectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=center_clustered(Nz, Lz, 0))

@info "  Testing diffusion cosine on ImmersedBoundaryGrid Stretched [$fieldname, $timestepper, $time_discretization]..."
immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> π/4))

κ, m = 1, 2 # diffusivity and cosine wavenumber

model = NonhydrostaticModel(timestepper = timestepper,
                                grid = immersed_grid,
                                closure = IsotropicDiffusivity(ν=κ, κ=κ, time_discretization=time_discretization),
                                tracers = (:T, :S),
                            buoyancy = nothing)

field = get_model_field(:T, model)

zC = znodes(Center, grid, reshape=true)

initfield= similar(field) 

interior(field)   .= cos.(m * zC)
interior(initfield) .= cos.(m * zC)

diffusing_cosine(κ, m, z, t) = exp(-κ * m^2 * t) * cos(m * z)

# Step forward with small time-step relative to diff. time-scale
Δt = 1e-6 * grid.Lz^2 / κ
for n in 1:1000
    ab2_or_rk3_time_step!(model, Δt, n)
end

half = Int(grid.Nz/2 + 1)

initial_half = interior(initfield)[1,1,half+1:end]
numerical_half = interior(field)[1,1,half+1:end]
analytical_half = diffusing_cosine.(κ, m, zC, model.clock.time)[1,1,half+1:end]

initial = interior(initfield)[1,1,:]
numerical = interior(field)[1,1,:]
analytical = diffusing_cosine.(κ, m, zC, model.clock.time)[1,1,:]

assessment = !any(@. !isapprox(numerical, analytical, atol=1e-6, rtol=1e-6))
