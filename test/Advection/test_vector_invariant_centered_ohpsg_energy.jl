using Test
using Oceananigans
using Oceananigans.Advection: VectorInvariant, U_dot_∇u, U_dot_∇v
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: covariant_to_contravariant_flux_uᶠᶜᶜ,
                              covariant_to_contravariant_flux_vᶜᶠᶜ,
                              hodge_weight_uᶠᶜᶜ,
                              hodge_weight_vᶜᶠᶜ,
                              horizontal_volume_flux_div_xyᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Random

function ohpsg_centered_vi_random_projected_model(; N = 4, seed = 42)
    grid = SphericalShellGrid(CPU(), Float64;
                              mapping = OctaHEALPixMapping(N),
                              z = (0, 1),
                              radius = 1,
                              halo = (5, 5, 3))

    model = HydrostaticFreeSurfaceModel(grid;
                                        tracers = (),
                                        buoyancy = nothing,
                                        coriolis = nothing,
                                        free_surface = nothing,
                                        closure = nothing,
                                        momentum_advection = VectorInvariant())

    u = model.velocities.u
    v = model.velocities.v

    Random.seed!(seed)
    fill!(parent(u), 0)
    fill!(parent(v), 0)

    for j in 1:grid.Ny, i in 1:grid.Nx
        u[i, j, 1] = convert(eltype(grid), 1//100) * randn()
        v[i, j, 1] = convert(eltype(grid), 1//100) * randn()
    end

    fill_halo_regions!((u, v))
    update_state!(model)

    return model
end

function centered_vi_hodge_work_defect(model)
    grid = model.grid
    scheme = model.advection.momentum
    velocities = model.velocities
    u = velocities.u
    v = velocities.v

    total_hodge_work = zero(eltype(grid))
    maximum_column_hodge_work = zero(eltype(grid))
    maximum_horizontal_divergence = zero(eltype(grid))
    maximum_column_hodge_work_index = 0

    for i in 1:grid.Nx
        column_hodge_work = zero(eltype(grid))

        for j in 1:grid.Ny
            u_tendency = -U_dot_∇u(i, j, 1, grid, scheme, velocities)
            v_tendency = -U_dot_∇v(i, j, 1, grid, scheme, velocities)
            u_flux = covariant_to_contravariant_flux_uᶠᶜᶜ(i, j, 1, grid, u, v)
            v_flux = covariant_to_contravariant_flux_vᶜᶠᶜ(i, j, 1, grid, u, v)

            column_hodge_work += u_tendency *
                                 hodge_weight_uᶠᶜᶜ(i, j, 1, grid) *
                                 u_flux

            column_hodge_work += v_tendency *
                                 hodge_weight_vᶜᶠᶜ(i, j, 1, grid) *
                                 v_flux

            divergence = horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, u, v)
            maximum_horizontal_divergence = max(maximum_horizontal_divergence, abs(divergence))
        end

        total_hodge_work += column_hodge_work

        if abs(column_hodge_work) > maximum_column_hodge_work
            maximum_column_hodge_work = abs(column_hodge_work)
            maximum_column_hodge_work_index = i
        end
    end

    return (; total_hodge_work,
              maximum_column_hodge_work,
              maximum_column_hodge_work_index,
              maximum_horizontal_divergence)
end

@testset "Centered VectorInvariant OHPSG Hodge skew-symmetry" begin
    model = ohpsg_centered_vi_random_projected_model()
    defect = centered_vi_hodge_work_defect(model)

    @info "Centered VI OHPSG Hodge skew-symmetry defect" defect.total_hodge_work defect.maximum_column_hodge_work defect.maximum_column_hodge_work_index defect.maximum_horizontal_divergence

    @test defect.maximum_horizontal_divergence < 1e-10
    @test_broken abs(defect.total_hodge_work) < 1e-12
end
