using Oceananigans
using Oceananigans.Simulations: reset!
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization
using Oceananigans.Advection: div_Uc
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using Oceananigans.Grids: get_active_cells_map
using Oceananigans.Operators: volume
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
# using Oceananigans: RightFaceFolded
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings
using Printf
using GLMakie

@info "Grid setup"

resolution = 2 // 1        # degrees
Nx = 360 ÷ resolution      # number of longitude points
Ny = 180 ÷ resolution + 1  # number of latitude points (avoiding poles)
# Nz = 50                    # number of vertical levels
# Nz = 75                    # number of vertical levels
Nz = 10
H = 5000                   # domain depth [m]
z = (-H, 0)                # vertical extent

underlying_tripolar_grid = TripolarGrid(
    size = (Nx, Ny, Nz),
    fold_topology = RightFaceFolded,
    z = z,
)

σφ, σλ = 5, 5     # mountain extent in latitude and longitude (degrees)
λ₀, φ₀ = 70, 55     # first pole location
h = H + 1000        # mountain height above the bottom (m)

gaussian(λ, φ) = exp(-((λ - λ₀)^2 / 2σλ^2 + (φ - φ₀)^2 / 2σφ^2))
gaussian_mountains(λ, φ) = (-H
    + h * (gaussian(λ, φ) + gaussian(λ - 180, φ) + gaussian(λ - 360, φ))
    + h/2 * (gaussian(λ - 90, 0) + gaussian(λ - 270, 0)) # extra seamounts
    + h/2 * (90 - φ) / 180 # slanted seafloor towards south pole
)

grid = ImmersedBoundaryGrid(underlying_tripolar_grid, GridFittedBottom(gaussian_mountains))

@info "Model setup"

# Instead of initializing with random velocities, infer them from a random initial streamfunction
# to ensure the velocity field is divergence-free at initialization.
ψ = Field{Face, Face, Center}(grid)
set!(ψ, rand(size(ψ)...))
velocities = PrescribedVelocityFields(; u = ∂y(ψ), v = -∂x(ψ))

@warn "Vertical closure must be explicit otherwise it won't be in the tendency!"
closure = (
    HorizontalScalarDiffusivity(κ = 300.0),
    VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = 1.0e-5),
    # VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ = 1.0e-5),
)

f0 = CenterField(grid, Real)

@warn "Adding newton_div method to allow sparsity tracer to pass through WENO"

ADTypes = Union{SparseConnectivityTracer.AbstractTracer, SparseConnectivityTracer.Dual, ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a, b::FT) where {FT <: ADTypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b) where {FT <: ADTypes} = a / b

model = HydrostaticFreeSurfaceModel(
    grid;
    velocities = velocities,
    tracer_advection = WENO(),
    # tracer_advection = Centered(order = 2),
    # tracer_advection = UpwindBiased(order = 1),
    tracers = (c = f0,),
    closure = closure,
)

@info "Functions to get vector of tendencies"

Nx′, Ny′, Nz′ = size(f0)
N = Nx′ * Ny′ * Nz′
fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
idx = findall(!isnan, interior(fNaN))
Nidx = length(idx)
@show N, Nidx
c0 = ones(Nidx)
c3D = zeros(Real, Nx′, Ny′, Nz′)
advection = model.advection[:c]
total_velocities = model.transport_velocities
kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
active_cells_map = get_active_cells_map(grid, Val(:interior))


@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

function mytendency(c)
    c3D[idx] .= c
    set!(model, c = c3D)
    c_tendency = CenterField(grid, Real)

    c_advection = model.advection[:c]
    c_forcing = model.forcing[:c]
    c_immersed_bc = immersed_boundary_condition(model.tracers[:c])

    args = tuple(
        Val(1),
        Val(:c),
        c_advection,
        model.closure,
        c_immersed_bc,
        model.buoyancy,
        model.biogeochemistry,
        model.transport_velocities,
        model.free_surface,
        model.tracers,
        model.closure_fields,
        model.auxiliary_fields,
        model.clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_Gc!,
        c_tendency,
        grid,
        args;
        active_cells_map
    )

    return interior(c_tendency)[idx]
end

@info "Autodiff setup"

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@info "Compute the Jacobian"

J = jacobian(mytendency, sparse_forward_backend, c0)

@info "Extra for vector of volumes"

@kernel function compute_volume!(vol, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds vol[i, j, k] = volume(i, j, k, grid, Center(), Center(), Center())
end

function compute_volume(grid)
    vol = CenterField(grid)
    (Nx, Ny, Nz) = size(vol)
    kernel_parameters = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    launch!(CPU(), grid, kernel_parameters, compute_volume!, vol, grid)
    return vol
end

volvec = interior(compute_volume(grid))[idx]

@info "Plot the Jacobian sparsity pattern"

fig, ax, plt = spy(
    0.5..size(J,1)+0.5,
    0.5..size(J,2)+0.5,
    J;
    # axis = (
    #     xticks = 1:12:size(J,1),
    #     yticks = 1:12:size(J,2)
    # ),
    colormap = :coolwarm,
    colorrange = maximum(abs.(J)) .* (-1, 1)
)
ylims!(ax, size(J,2)+0.5, 0.5)
Colorbar(fig[1, 2], plt)
fig

# TODO: Figure out a way to check the correctness of the jacobian = transport matrix!
# The tendency is F(c) = ∂c/∂t, and the Jacobian is J(c) = ∂F/∂c(c) = ∂(∂c/∂t)/∂c.
# If F is linear in c, then we have that
# F(c) = J c
# So we can check that J c0 ≈ F(c0) = mytendency(c0)

@info "Assert Jacobian is correct for linear tendency"

@assert J * c0 ≈ mytendency(c0)