using CUDA
using Oceananigans.Solvers: solve!, set_source_term!
using Oceananigans.Solvers: poisson_eigenvalues
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _compute_w_from_continuity!
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

function poisson_solver_instantiates(grid, planner_flag)
    solver = FFTBasedPoissonSolver(grid, planner_flag)
    return true  # Just making sure the FFTBasedPoissonSolver does not error/crash.
end

function random_divergent_source_term(grid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()
    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    Ru = CenterField(grid, boundary_conditions=u_bcs)
    Rv = CenterField(grid, boundary_conditions=v_bcs)
    Rw = CenterField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R, U
end

function random_divergence_free_source_term(grid)
    default_bcs = FieldBoundaryConditions()
    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    # Random right hand side
    Ru = CenterField(grid, boundary_conditions=u_bcs)
    Rv = CenterField(grid, boundary_conditions=v_bcs)
    Rw = CenterField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, zeros(Nx, Ny, Nz))

    arch = architecture(grid)
    fill_halo_regions!(Ru, nothing, nothing)
    fill_halo_regions!(Rv, nothing, nothing)
    fill_halo_regions!(Rw, nothing, nothing)

    event = launch!(arch, grid, :xy, _compute_w_from_continuity!, U, grid,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(Rw, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, Ru.data, Rv.data, Rw.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    return R
end

#####
##### Regular rectilinear grid Poisson solver
#####

function divergence_free_poisson_solution(grid, planner_flag=FFTW.MEASURE)
    arch = architecture(grid)
    ArrayType = array_type(arch)
    FT = eltype(grid)

    solver = FFTBasedPoissonSolver(grid, planner_flag)
    R, U = random_divergent_source_term(grid)

    p_bcs = FieldBoundaryConditions(grid, (Center, Center, Center))
    ϕ   = CenterField(grid, boundary_conditions=p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(grid, boundary_conditions=p_bcs)

    # Using Δt = 1 but it doesn't matter since velocities = 0.
    solve_for_pressure!(ϕ.data, solver, 1, U)

    compute_∇²!(∇²ϕ, ϕ, arch, grid)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Test that Poisson solver error converges as error ~ N⁻²
#####

ψ(::Type{Bounded}, n, x) = cos(n*x/2)
ψ(::Type{Periodic}, n, x) = cos(n*x)

k²(::Type{Bounded}, n) = (n/2)^2
k²(::Type{Periodic}, n) = n^2

function analytical_poisson_solver_test(arch, N, topo; FT=Float64, mode=1)
    grid = RectilinearGrid(arch, FT, topology=topo, size=(N, N, N), x=(0, 2π), y=(0, 2π), z=(0, 2π))
    solver = FFTBasedPoissonSolver(grid)

    xC, yC, zC = nodes((Center, Center, Center), grid, reshape=true)

    TX, TY, TZ = topology(grid)
    Ψ(x, y, z) = ψ(TX, mode, x) * ψ(TY, mode, y) * ψ(TZ, mode, z)
    f(x, y, z) = -(k²(TX, mode) + k²(TY, mode) + k²(TZ, mode)) * Ψ(x, y, z)

    solver.storage .= convert(array_type(arch), f.(xC, yC, zC))

    ϕc = rhs = solver.storage
    solve!(ϕc, solver, rhs)

    ϕ = real(Array(solver.storage))

    L¹_error = mean(abs, ϕ - Ψ.(xC, yC, zC))

    return L¹_error
end

function poisson_solver_convergence(arch, topo, N¹, N²; FT=Float64, mode=1)
    error¹ = analytical_poisson_solver_test(arch, N¹, topo; FT, mode)
    error² = analytical_poisson_solver_test(arch, N², topo; FT, mode)

    rate = log(error¹ / error²) / log(N² / N¹)

    TX, TY, TZ = topo
    @info "Convergence of L¹-normed error, $(typeof(arch)), $FT, ($(N¹)³ -> $(N²)³), topology=($TX, $TY, $TZ): $rate"

    return isapprox(rate, 2, rtol=5e-3)
end

#####
##### Vertically stretched Poisson solver
#####

get_grid_size(TX, TY, TZ, Nx, Ny, Nz) = (Nx, Ny, Nz)
get_grid_size(::Type{Flat}, TY, TZ, Nx, Ny, Nz) = (Ny, Nz)
get_grid_size(TX, ::Type{Flat}, TZ, Nx, Ny, Nz) = (Nx, Nz)

get_xy_interval_kwargs(TX, TY, TZ) = (x=(0, 1), y=(0, 1))
get_xy_interval_kwargs(TX, ::Type{Flat}, TZ) = (x=(0, 1),)
get_xy_interval_kwargs(::Type{Flat}, TY, TZ) = (y=(0, 1),)

function vertically_stretched_poisson_solver_correct_answer(FT, arch, topo, Nx, Ny, zF)
    Nz = length(zF) - 1
    sz = get_grid_size(topo..., Nx, Ny, Nz)
    xy_intervals = get_xy_interval_kwargs(topo...)
    vs_grid = RectilinearGrid(arch, FT; topology=topo, size=sz, z=zF, xy_intervals...)
    solver = FourierTridiagonalPoissonSolver(vs_grid)

    p_bcs = FieldBoundaryConditions(vs_grid, (Center, Center, Center))
    ϕ   = CenterField(vs_grid, boundary_conditions=p_bcs)  # "kinematic pressure"
    ∇²ϕ = CenterField(vs_grid, boundary_conditions=p_bcs)

    R = random_divergence_free_source_term(vs_grid)

    set_source_term!(solver, R)
    ϕc = solver.storage
    solve!(ϕc, solver)

    # interior(ϕ) = solution(solver) or solution!(interior(ϕ), solver)
    CUDA.@allowscalar interior(ϕ) .= real.(solver.storage)
    compute_∇²!(∇²ϕ, ϕ, arch, vs_grid)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end
