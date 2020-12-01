using Oceananigans.Solvers: solve_for_pressure!, solve_poisson_equation!

function pressure_solver_instantiates(arch, FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    solver = PressureSolver(arch, grid, planner_flag)
    return true  # Just making sure the PressureSolver does not error/crash.
end

function divergence_free_poisson_solution(arch, FT, topo, Nx, Ny, Nz, planner_flag=FFTW.MEASURE)
    ArrayType = array_type(arch)
    grid = RegularCartesianGrid(FT, topology=topo, size=(Nx, Ny, Nz), extent=(1.0, 2.5, π))
    p_bcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, planner_flag)

    # Generate right hand side from a random (divergent) velocity field.
    Ru = CellField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CellField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CellField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    # Adding (nothing, nothing) in case we need to dispatch on ::NFBC
    fill_halo_regions!(Ru, arch, nothing, nothing)
    fill_halo_regions!(Rv, arch, nothing, nothing)
    fill_halo_regions!(Rw, arch, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    ϕ   = CellField(FT, arch, grid, p_bcs)  # "pressure"
    ∇²ϕ = CellField(FT, arch, grid, p_bcs)

    # Using Δt = 1 but it doesn't matter since velocities = 0.
    solve_for_pressure!(ϕ.data, solver, arch, grid, 1, datatuple(U))

    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ.data, ∇²ϕ.data, dependencies=Event(device(arch)))
    wait(device(arch), event)
    fill_halo_regions!(∇²ϕ, arch)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Test that Poisson solver error converges as error ~ N⁻²
#####

ψ(::Type{Bounded}, n, x) = cos(n*x/2)
ψ(::Type{Periodic}, n, x) = cos(n*x)

k²(::Type{Bounded}, n) = (n/2)^2
k²(::Type{Periodic}, n) = n^2

using Oceananigans.Solvers: permute_index, unpermute_index

@kernel function permute_indices!(dst, src, solver_type, arch, grid)
    i, j, k = @index(Global, NTuple)

    i′, j′, k′ = permute_index(solver_type, arch, i, j, k, grid.Nx, grid.Ny, grid.Nz)

    @inbounds dst[i′, j′, k′] = src[i, j, k]
end

@kernel function unpermute_indices!(dst, src, solver_type, arch, grid)
    i, j, k = @index(Global, NTuple)

    i′, j′, k′ = unpermute_index(solver_type, arch, i, j, k, grid.Nx, grid.Ny, grid.Nz)

    @inbounds dst[i′, j′, k′] = src[i, j, k]
end

function analytical_poisson_solver_test(arch, N, topo; FT=Float64, mode=1)
    grid = RegularCartesianGrid(FT, topology=topo, size=(N, N, N), x=(0, 2π), y=(0, 2π), z=(0, 2π))
    solver = PressureSolver(arch, grid, TracerBoundaryConditions(grid))

    xC, yC, zC = nodes((Cell, Cell, Cell), grid, reshape=true)

    Tx, Ty, Tz = topology(grid)
    Ψ(x, y, z) = ψ(Tx, mode, x) * ψ(Ty, mode, y) * ψ(Tz, mode, z)
    f(x, y, z) = -(k²(Tx, mode) + k²(Ty, mode) + k²(Tz, mode)) * Ψ(x, y, z)

    if arch isa GPU && topo == PBB_topo
        storage = solver.storage.storage1
    else
        storage = solver.storage
    end

    buffer = similar(storage)
    buffer .= convert(array_type(arch), f.(xC, yC, zC))

    event = launch!(arch, grid, :xyz,
                    permute_indices!, storage, buffer, solver.type, arch, grid,
                    dependencies = Event(device(arch)))
    wait(device(arch), event)

    solve_poisson_equation!(solver, grid)

    event = launch!(arch, grid, :xyz,
                    unpermute_indices!, buffer, storage, solver.type, arch, grid,
                    dependencies = Event(device(arch)))
    wait(device(arch), event)

    ϕ = real(Array(buffer))

    L¹_error = mean(abs, ϕ - Ψ.(xC, yC, zC))

    return L¹_error
end

function poisson_solver_convergence(arch, topo, N¹, N²; FT=Float64)
    error¹ = analytical_poisson_solver_test(arch, N¹, topo; FT=FT)
    error² = analytical_poisson_solver_test(arch, N², topo; FT=FT)

    rate = log(error¹ / error²) / log(N² / N¹)

    Tx, Ty, Tz = topo
    @info "Convergence of L¹-normed error, $(typeof(arch)), $FT, ($(N¹)³ -> $(N²)³), topology=($Tx, $Ty, $Tz): $rate"

    return isapprox(rate, 2, rtol=5e-3)
end

#####
##### Run pressure solver tests
#####

const PPP_topo = (Periodic, Periodic, Periodic)
const PPB_topo = (Periodic, Periodic, Bounded)
const PBB_topo = (Periodic, Bounded,  Bounded)
const BBB_topo = (Bounded,  Bounded,  Bounded)

const PBP_topo = (Periodic, Bounded, Periodic)

topos = (PPP_topo, PPB_topo, PBB_topo, BBB_topo)

@testset "Pressure solvers" begin
    @info "Testing pressure solvers..."

    PB = (Periodic, Bounded)
    all_topos = collect(Iterators.product(PB, PB, PB))[:]

    # for arch in archs
    #     @testset "Pressure solver instantiation [$(typeof(arch))]" begin
    #         @info "  Testing pressure solver instantiation [$(typeof(arch))]..."
    #         for FT in float_types
    #             @test pressure_solver_instantiates(arch, FT, 32, 32, 32, FFTW.ESTIMATE)
    #             @test pressure_solver_instantiates(arch, FT, 1,  32, 32, FFTW.MEASURE)
    #             @test pressure_solver_instantiates(arch, FT, 32,  1, 32, FFTW.ESTIMATE)
    #             @test pressure_solver_instantiates(arch, FT,  1,  1, 32, FFTW.MEASURE)
    #         end
    #     end
    # end

    @test divergence_free_poisson_solution(CPU(), Float64, PPP_topo, 16, 16, 16, FFTW.ESTIMATE)
    @test divergence_free_poisson_solution(CPU(), Float64, PPB_topo, 16, 16, 16, FFTW.ESTIMATE)
    @test divergence_free_poisson_solution(CPU(), Float64, PBB_topo, 16, 16, 16, FFTW.ESTIMATE)
    @test divergence_free_poisson_solution(CPU(), Float64, BBB_topo, 16, 16, 16, FFTW.ESTIMATE)

    @test divergence_free_poisson_solution(GPU(), Float64, PPP_topo, 16, 16, 16, FFTW.ESTIMATE)
    @test divergence_free_poisson_solution(GPU(), Float64, PPB_topo, 16, 16, 16, FFTW.ESTIMATE)
    @test divergence_free_poisson_solution(GPU(), Float64, PBB_topo, 16, 16, 16, FFTW.ESTIMATE)
    # @test divergence_free_poisson_solution(GPU(), Float64, PBP_topo, 16, 16, 16, FFTW.ESTIMATE)

    #=
    @testset "Divergence-free solution [CPU]" begin
        @info "  Testing divergence-free solution [CPU]..."

        for topo in all_topos
            @info "    Testing $topo topology on square grids..."
            for N in [7, 16], FT in float_types
                @test divergence_free_poisson_solution(CPU(), FT, topo, N, N, N, FFTW.ESTIMATE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, 1, N, N, FFTW.MEASURE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, N, 1, N, FFTW.ESTIMATE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, 1, 1, N, FFTW.MEASURE)
            end
        end

        Ns = [11, 16]
        for topo in all_topos
            @info "    Testing $topo topology on rectangular grids..."
            for Nx in Ns, Ny in Ns, Nz in Ns, FT in float_types
                @test divergence_free_poisson_solution(CPU(), FT, topo, Nx, Ny, Nz, FFTW.ESTIMATE)
            end
        end
    end

    @hascuda @testset "Divergence-free solution [GPU]" begin
        @info "  Testing divergence-free solution [GPU]..."
        for topo in (PPP_topo, PPB_topo, PBB_topo)
            @info "    Testing $topo topology on GPUs..."
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 16, 16, 16)
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 32, 32, 16)
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 16, 32, 24)
	        @test divergence_free_poisson_solution(GPU(), Float64, topo, 5,  7,  11)
        end
    end

    @testset "Convergence to analytical solution [CPU]" begin
        @info "  Testing convergence to analytical solution [CPU]..."
        for topo in topos
            @test poisson_solver_convergence(CPU(), topo, 2^6, 2^7)
            @test poisson_solver_convergence(CPU(), topo, 67, 131)
        end
    end

    @hascuda @testset "Convergence to analytical solution [GPU]" begin
        @info "  Testing convergence to analytical solution [GPU]..."
        for topo in (PPP_topo, PPB_topo, PBB_topo)
            @test poisson_solver_convergence(GPU(), topo, 2^6, 2^7)
            @test poisson_solver_convergence(GPU(), topo, 67, 131)
        end
    end
    =#
end
