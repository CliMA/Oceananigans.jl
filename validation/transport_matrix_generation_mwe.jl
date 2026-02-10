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

resolution = 4 // 1        # degrees
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

# f0 = CenterField(grid, Real)
f0 = CenterField(grid)

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
    # tracer_advection = WENO(),
    tracer_advection = Centered(order = 2),
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
kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
active_cells_map = get_active_cells_map(grid, Val(:interior))


@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end


function mytendency(cvec)
    Gcvec = similar(cvec)
    mytendency!(Gcvec, cvec)
    return Gcvec
end

function mytendency!(Gcvec::Vector{T}, cvec::Vector{T}) where {T}
    # Preallocate 3D array with type T and fill wet points
    c3D = zeros(T, Nx′, Ny′, Nz′)
    c3D[idx] .= cvec
    # Preallocate Field with type T and fill it with 3D array
    c = CenterField(grid, T)
    set!(c, c3D)
    # Preallocate "output" Field with type T
    Gc = CenterField(grid, T)
    # bits and pieces from model
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
        (; c = c),
        model.closure_fields,
        model.auxiliary_fields,
        model.clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_Gc!,
        Gc,
        grid,
        args;
        active_cells_map
    )
    # Fill output vector with interior wet values
    Gcvec .= interior(Gc)[idx]
    return Gcvec
end

# function myfieldtendency(c)
#     dc = similar(c)
#     myfieldtendency!(dc, c)
#     return dc
# end

# function myfieldtendency!(dc, c)


#     c_advection = model.advection[:c]
#     c_forcing = model.forcing[:c]
#     c_immersed_bc = immersed_boundary_condition(model.tracers[:c])

#     args = tuple(
#         Val(1),
#         Val(:c),
#         c_advection,
#         model.closure,
#         c_immersed_bc,
#         model.buoyancy,
#         model.biogeochemistry,
#         model.transport_velocities,
#         model.free_surface,
#         (c = c,),
#         model.closure_fields,
#         model.auxiliary_fields,
#         model.clock,
#         c_forcing
#     )

#     launch!(
#         CPU(), grid, kernel_parameters,
#         compute_hydrostatic_free_surface_Gc!,
#         dc,
#         grid,
#         args;
#         active_cells_map
#     )

#     return dc
# end

# # foo

@info "Autodiff setup"

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@info "Compute the Jacobian"
using BenchmarkTools

# c = CenterField(grid)
# dc = CenterField(grid)

# J = jacobian(myfieldtendency, sparse_forward_backend, c)

# @benchmark jacobian(
#     $myfieldtendency, $sparse_forward_backend, $c
# )

# jac_prep_sparse = prepare_jacobian(myfieldtendency, sparse_forward_backend, c)

# @benchmark jacobian(
#     $myfieldtendency, $jac_prep_sparse, $sparse_forward_backend, $c
# )

# jac_buffer = similar(sparsity_pattern(jac_prep_sparse), eltype(c))

# @benchmark jacobian!(
#     $myfieldtendency, $jac_buffer, $jac_prep_sparse, $sparse_forward_backend, $c
# )

# jac_prep_sparse_nonallocating = prepare_jacobian(
#     myfieldtendency!, dc, sparse_forward_backend, c
# )

# jac_buffer = similar(sparsity_pattern(jac_prep_sparse_nonallocating), eltype(c))

# @benchmark jacobian!(
#     myfieldtendency!,
#     $dc,
#     $jac_buffer,
#     $jac_prep_sparse_nonallocating,
#     $sparse_forward_backend,
#     $c,
# )

# foo

J = jacobian(mytendency, sparse_forward_backend, c0)
dc0 = similar(c0)

jac_prep_sparse = prepare_jacobian(mytendency!, dc0, sparse_forward_backend, c0; strict=Val(false))
jac_buffer = similar(sparsity_pattern(jac_prep_sparse), eltype(c0))
@benchmark jacobian!(
    mytendency!,
    $dc0,
    $jac_buffer,
    $jac_prep_sparse,
    $sparse_forward_backend,
    $c0,
)

DualType = eltype(DifferentiationInterface.overloaded_input_type(jac_prep_sparse))
# Preallocate 3D array with type T and fill wet points
c3D_dual = zeros(DualType, Nx′, Ny′, Nz′)
# Preallocate Field with type T and fill it with 3D array
c_dual = CenterField(grid, DualType)
# Preallocate "output" Field with type T
Gc_dual = CenterField(grid, DualType)

function mytendency_preallocated!(Gcvec::Vector{DualType}, cvec::Vector{DualType})
    c3D_dual[idx] .= cvec
    set!(c_dual, c3D_dual)
    # bits and pieces from model
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
        (; c = c_dual),
        model.closure_fields,
        model.auxiliary_fields,
        model.clock,
        c_forcing
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_Gc!,
        Gc_dual,
        grid,
        args;
        active_cells_map
    )
    # Fill output vector with interior wet values
    Gcvec .= view(interior(Gc_dual), idx)
    return Gcvec
end

@benchmark jacobian!(
    mytendency_preallocated!,
    $dc0,
    $jac_buffer,
    $jac_prep_sparse,
    $sparse_forward_backend,
    $c0,
)

J2 = jacobian!(
    mytendency_preallocated!,
    dc0,
    jac_buffer,
    jac_prep_sparse,
    sparse_forward_backend,
    c0,
)

@assert J == J2

# # @benchmark jacobian($mytendency, $sparse_forward_backend, $c0)
# # @benchmark jacobian($mytendency!, $dc0, $sparse_forward_backend, $c0)


# # @info "Extra for vector of volumes"

# # @kernel function compute_volume!(vol, grid)
# #     i, j, k = @index(Global, NTuple)
# #     @inbounds vol[i, j, k] = volume(i, j, k, grid, Center(), Center(), Center())
# # end

# # function compute_volume(grid)
# #     vol = CenterField(grid)
# #     (Nx, Ny, Nz) = size(vol)
# #     kernel_parameters = KernelParameters(1:Nx, 1:Ny, 1:Nz)
# #     launch!(CPU(), grid, kernel_parameters, compute_volume!, vol, grid)
# #     return vol
# # end

# # volvec = interior(compute_volume(grid))[idx]

# # @info "Plot the Jacobian sparsity pattern"

# # fig, ax, plt = spy(
# #     0.5..size(J,1)+0.5,
# #     0.5..size(J,2)+0.5,
# #     J;
# #     # axis = (
# #     #     xticks = 1:12:size(J,1),
# #     #     yticks = 1:12:size(J,2)
# #     # ),
# #     colormap = :coolwarm,
# #     colorrange = maximum(abs.(J)) .* (-1, 1)
# # )
# # ylims!(ax, size(J,2)+0.5, 0.5)
# # Colorbar(fig[1, 2], plt)
# # fig

# # TODO: Figure out a way to check the correctness of the jacobian = transport matrix!
# # The tendency is F(c) = ∂c/∂t, and the Jacobian is J(c) = ∂F/∂c(c) = ∂(∂c/∂t)/∂c.
# # If F is linear in c, then we have that
# # F(c) = J c
# # So we can check that J c0 ≈ F(c0) = mytendency(c0)

# @info "Assert Jacobian is correct for linear tendency"

# @info "profiling now"

# @assert J * c0 ≈ mytendency(c0)

# using Profile
# using PProf
# Profile.clear()
# @profile jacobian!(
#     mytendency!,
#     dc0,
#     jac_buffer,
#     jac_prep_sparse,
#     sparse_forward_backend,
#     c0,
# )
# Profile.clear()
# @profile jacobian!(
#     mytendency!,
#     dc0,
#     jac_buffer,
#     jac_prep_sparse,
#     sparse_forward_backend,
#     c0,
# )
# pprof()



z1D = reshape(znodes(grid, Center(), Center(), Center()), 1, 1, Nz)
srf = z1D .≥ z1D[Nz] * ones(Nx, Ny)
using SparseArrays
using LinearAlgebra
L = sparse(Diagonal(srf[idx]))
M = J - L
age = M \ -ones(Nidx)
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
fig, ax, plt = hist(age / year)