using Statistics

using Oceananigans.Grids: halo_size

#=
function run_horizontal_average_tests(arch, FT)
    topo = (Periodic, Periodic, Bounded)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    Hx, Hy, Hz = halo_size(grid)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    linear(x, y, z) = z
    set!(model, T=linear, w=linear)

    T̅ = Average(model.tracers.T, dims=(1, 2), with_halos=false)
    computed_T_profile = T̅(model)
    @test size(computed_T_profile) == (1, 1, Nz)
    @test computed_T_profile ≈ znodes(Cell, grid, reshape=true)

    T̅ = Average(model.tracers.T, dims=(1, 2), with_halos=true)
    computed_T_profile_with_halos = T̅(model)
    @test size(computed_T_profile_with_halos) == (1, 1, Nz+2Hz)
    @test computed_T_profile_with_halos[1+Hz:Nz+Hz] ≈ znodes(Cell, grid)

    w̅ = Average(model.velocities.w, dims=(1, 2), with_halos=false)
    computed_w_profile = w̅(model)
    @test size(computed_w_profile) == (1, 1, Nz+1)
    @test computed_w_profile ≈ znodes(Face, grid, reshape=true)

    w̅ = Average(model.velocities.w, dims=(1, 2), with_halos=true)
    computed_w_profile_with_halos = w̅(model)
    @test size(computed_w_profile_with_halos) == (1, 1, Nz+1+2Hz)
    @test computed_w_profile_with_halos[1+Hz:Nz+1+Hz] ≈ znodes(Face, grid)
end

function run_zonal_average_tests(arch, FT)
    topo = (Periodic, Bounded, Bounded)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    Hx, Hy, Hz = halo_size(grid)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    linear(x, y, z) = z
    set!(model, T=linear, v=linear)

    T̅ = Average(model.tracers.T, dims=1, with_halos=false)
    computed_T_slice = T̅(model)
    @test size(computed_T_slice) == (1, Ny, Nz)

    computed_T_slice = dropdims(computed_T_slice, dims=1)
    zC = znodes(Cell, grid)
    @test all(computed_T_slice[j, :] ≈ zC for j in 1:Ny)

    T̅ = Average(model.tracers.T, dims=1, with_halos=true)
    computed_T_slice_with_halos = T̅(model)
    @test size(computed_T_slice_with_halos) == (1, Ny+2Hy, Nz+2Hz)

    computed_T_slice_with_halos = dropdims(computed_T_slice_with_halos, dims=1)
    @test computed_T_slice_with_halos[1+Hy:Ny+Hy, 1+Hz:Nz+Hz] ≈ computed_T_slice

    v̅ = Average(model.velocities.v, dims=1, with_halos=false)
    computed_v_slice = v̅(model)
    @test size(computed_v_slice) == (1, Ny+1, Nz)

    computed_v_slice = dropdims(computed_v_slice, dims=1)
    zC = znodes(Cell, grid)
    @test all(computed_v_slice[j, :] ≈ zC for j in 1:Ny)

    v̅ = Average(model.velocities.v, dims=1, with_halos=true)
    computed_v_slice_with_halos = v̅(model)
    @test size(computed_v_slice_with_halos) == (1, Ny+1+2Hy, Nz+2Hz)

    computed_v_slice_with_halos = dropdims(computed_v_slice_with_halos, dims=1)
    @test computed_v_slice_with_halos[1+Hy:Ny+1+Hy, 1+Hz:Nz+Hz] ≈ computed_v_slice
end

function run_volume_average_tests(arch, FT)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(100, 100, 100))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = z
    set!(model, T=T₀)

    T̅ = Average(model.tracers.T, dims=(1, 2, 3), time_interval=0.5second, with_halos=false)
    computed_scalar = T̅(model)
    @test size(computed_scalar) == (1, 1, 1)
    @test all(computed_scalar .≈ -50.0)

    T̅ = Average(model.tracers.T, dims=(1, 2, 3), time_interval=0.5second, with_halos=true)
    computed_scalar_with_halos = T̅(model)
    @test size(computed_scalar_with_halos) == (1, 1, 1)
    @test all(computed_scalar_with_halos .≈ -50.0)
end

TestModel(::GPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(16, 16, 16), extent=(16Δx, 16Δx, 16Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = GPU(),
    float_type = FT
)

TestModel(::CPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = CPU(),
    float_type = FT
)
=#

@testset "Averaged fields" begin
    @info "Testing averaged fields..."

    for arch in archs
        @testset "Averaged fields [$(typeof(arch))]" begin
            @info "  Testing AveragedFields [$(typeof(arch))]"
            for FT in float_types

                grid = RegularCartesianGrid(topology=(Periodic, Periodic, Bounded), size=(2, 2, 2),
                                            x=(0, 2), y=(0, 2), z=(0, 2))
                model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

                u, v, w = model.velocities
                T, S = model.tracers

                trilinear(x, y, z) = x + y + z
                set!(model, T=trilinear, w=trilinear)

                T̃ = mean(T, dims=(1, 2, 3))
                T̅ = mean(T, dims=(1, 2))
                T̂ = mean(T, dims=1)

                w̃ = mean(w, dims=(1, 2, 3))
                w̅ = mean(w, dims=(1, 2))
                ŵ = mean(w, dims=1)

                for ϕ in (T̃, T̅, T̂, w̃, w̅, ŵ)
                    compute!(ϕ)
                end

                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

                @test T̂[1, 1, 1] ≈ 3
                @test Array(T̅[1, 1, 1:Nz]) ≈ [2.5, 3.5]
                @test Array(T̃[1, 1:Ny, 1:Nz]) ≈ [[2, 3] [3, 4]]

                @test ŵ[1, 1, 1] ≈ 4
                @test Array(w̅[1, 1, 1:Nz+1]) ≈ [2, 4, 6]
                @test Array(w̃[1, 1:Ny, 1:Nz]) ≈ [[2, 3] [3, 4]]

                #@test T̃[1, 1, 1:Nz] ≈ znodes(Cell, grid, reshape=true)
                #@test T̂[1, 1, 1:Nz] ≈ znodes(Cell, grid, reshape=true)

                @test w̅[1, 1, 1:Nz] ≈ znodes(Face, grid, reshape=true)
                @test w̅[1, 1, 1:Nz] ≈ znodes(Face, grid, reshape=true)
            end
        end
    end
end
