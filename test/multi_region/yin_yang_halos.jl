include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.MultiRegion: YinYangInterpolationTable, yin_yang_interpolation, interpolation_table,
                                stencil_start, lagrange_weight

# Geographic velocity of a wavenumber-4 Rossby-Haurwitz wave
rossby_haurwitz_velocity(λ, φ; R = 4) = (cosd(φ) + cosd(φ)^(R-1) * (R * sind(φ)^2 - cosd(φ)^2) * cosd(R * λ),
                                         -R * cosd(φ)^(R-1) * sind(φ) * sind(R * λ))

function scalar_halo_error(grid, ℓx, ℓy, f = spherical_harmonics)
    c = location_field(ℓx, ℓy, grid)
    set!(c, (λ, φ, z) -> f(λ, φ))
    fill_halo_regions!(c)
    return panel_error(c, (λ, φ, r) -> f(λ, φ))
end

function vector_halo_error(grid, velocity)
    u, v = XFaceField(grid), YFaceField(grid)
    set_panel_components!(u, v, velocity)
    fill_halo_regions!((u, v))
    return max(panel_error(u, panel_u(velocity, grid)), panel_error(v, panel_v(velocity, grid)))
end

@testset "Yin-Yang halo filling [$(typeof(arch))]" for arch in archs
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0))
    typed_grid = YinYangGrid(arch, Float32; size=(48, 16, 1), z=(-1, 0))
    Nx, Ny, _ = size(grid)

    @testset "Interpolation tables" begin
        interpolation = yin_yang_interpolation(grid)
        receivers = halo_nodes(grid)
        N = length(receivers)
        panels = (on_architecture(CPU(), grid[1]), on_architecture(CPU(), grid[2]))

        # Every table lists the halo nodes of either panel, with a stencil of interior donors of the other panel
        # that brackets the node, and the weights and rotation angle of the node's position in the donor panel
        for (ℓʳ, ℓᵈ, name) in (((Center(), Center()), (Center(), Center()), :center_center),
                               ((Face(), Face()), (Face(), Face()), :face_face),
                               ((Face(), Center()), (Center(), Face()), :face_center_from_center_face)),
            (receiver, donor) in ((1, 2), (2, 1))

            table = on_architecture(CPU(), getproperty(interpolation, name))
            @test table isa YinYangInterpolationTable{4}
            @test collect(zip(table.receivers.i, table.receivers.j)) == receivers
            @test all(1 .≤ table.donors.i .≤ Nx - 3) && all(1 .≤ table.donors.j .≤ Ny - 3)

            λφ = [node_coordinates(i, j, panels[receiver], ℓʳ...) for (i, j) in receivers]
            indices = [geographic_fractional_indices(λ, φ, grid, donor, ℓᵈ...) for (λ, φ) in λφ]
            i★, j★ = first.(indices), last.(indices)
            poles = north_pole(panels[receiver]), north_pole(panels[donor])
            ψ = [rotated_basis_angle(λ, φ, poles[1]...) - rotated_basis_angle(λ, φ, poles[2]...) for (λ, φ) in λφ]
            @test maximum(abs.(stencil_start.(i★, 4) .- table.donors.i)) ≤ 1
            @test maximum(abs.(stencil_start.(j★, 4) .- table.donors.j)) ≤ 1
            @test table.weights.x ≈ [lagrange_weight(i★[n], table.donors.i[n], 4, a) for n in 1:N, a in 1:4] rtol=1e-6
            @test table.weights.y ≈ [lagrange_weight(j★[n], table.donors.j[n], 4, a) for n in 1:N, a in 1:4] rtol=1e-6
            @test table.rotation.cosψ ≈ cos.(ψ) rtol=1e-10
            @test table.rotation.sinψ ≈ sin.(ψ) rtol=1e-10
        end

        @test interpolation_table(interpolation, Center(), Center()) === interpolation.center_center
        @test interpolation_table(interpolation, Face(), Center(), Center(), Face()) === interpolation.face_center_from_center_face
        @test eltype(yin_yang_interpolation(typed_grid).face_center.weights.x) == Float32
    end

    @testset "Scalar halo fills" begin
        errors = [scalar_halo_error(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)), Center(), Center())
                  for Nφ in (16, 32)]
        @test errors[1] < 1e-3
        @test only(convergence_rates(errors)) > 3.5
        @test all(scalar_halo_error(grid, ℓx, ℓy) < 1e-3 for (ℓx, ℓy) in yin_yang_locations)
        @test scalar_halo_error(typed_grid, Face(), Center()) < 1e-3

        bilinear = YinYangGrid(arch; size=(48, 16, 1), z=(-1, 0), stencil_width=2)
        @test yin_yang_interpolation(bilinear).center_center isa YinYangInterpolationTable{2}
        @test errors[1] < scalar_halo_error(bilinear, Center(), Center()) < 1e-1

        # The fill reaches the vertical halos, and ignores the extra arguments and keywords
        c = CenterField(grid)
        set!(c, (λ, φ, z) -> spherical_harmonics(λ, φ))
        fill_halo_regions!(c, nothing, nothing; async=true, only_local_halos=true)
        for r in 1:2
            data = on_architecture(CPU(), getregion(c, r))
            @test data[1, 1, 0] == data[1, 1, 1]
            @test data[-2, 25, 3] == data[-2, 25, 2]
        end
        @test c.communication_buffers === nothing

        # Fields windowed in x or y get only their vertical halos
        for indices in ((1:10, :, :), (:, 1:5, :), (:, :, 2:2))
            d = CenterField(grid; indices)
            set!(d, 1)
            fill_halo_regions!(d)
            @test all(==(1), interior(getregion(d, 2)))
        end
    end

    @testset "Vector halo fills" begin
        @test vector_halo_error(grid, solid_body_velocity) < 1e-5
        @test vector_halo_error(grid, rossby_haurwitz_velocity) < 2e-3

        # Fields without boundary conditions have no halos to fill
        u, v = XFaceField(grid; boundary_conditions=nothing), YFaceField(grid; boundary_conditions=nothing)
        @test fill_halo_regions!((u, v)) === nothing
    end
end
