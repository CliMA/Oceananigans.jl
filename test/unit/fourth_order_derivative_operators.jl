include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Operators: fourth_order_∂xᵣᶠᶜᶜ, fourth_order_∂xᵣᶜᶜᶜ, fourth_order_∂yᵣᶜᶠᶜ, fourth_order_∂zᶜᶜᶠ,
                              fourth_order_∂zᶠᶜᶠ, fourth_order_δxᶠᵃᵃ, fourth_order_x_stencil, fourth_order_z_stencil,
                              ∂xᵣᶠᶜᶜ, ∂xᵣᶜᶜᶜ, ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂zᶜᶜᶜ, ∂zᶜᶜᶠ, ∂zᶠᶜᶠ

convergence_rate(e₁, e₂) = log2(e₁ / e₂)

# Flux divergence of the fourth-order-accurate first derivative: the fourth-order five-point Laplacian
@inline ∂x_∂x⁽⁴⁾(i, j, k, grid, c) = ∂xᶜᶜᶜ(i, j, k, grid, fourth_order_∂xᵣᶠᶜᶜ, c)
@inline ∂y_∂y⁽⁴⁾(i, j, k, grid, c) = ∂yᶜᶜᶜ(i, j, k, grid, fourth_order_∂yᵣᶜᶠᶜ, c)
@inline ∂z_∂z⁽⁴⁾(i, j, k, grid, c) = ∂zᶜᶜᶜ(i, j, k, grid, fourth_order_∂zᶜᶜᶠ, c)

function second_derivative_errors(arch, ∂², N)
    grid = RectilinearGrid(arch; size=(N, N, N), extent=(2π, 2π, 2π), halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))
    c = CenterField(grid)
    set!(c, (x, y, z) -> sin(x) * sin(y) * sin(z))
    fill_halo_regions!(c)
    error = Field(KernelFunctionOperation{Center, Center, Center}(∂², grid, c) + c) # ∂²c = - c
    return maximum(abs, error)
end

function first_derivative_error(arch, N)
    grid = RectilinearGrid(arch; size=N, x=(0, 2π), halo=3, topology=(Periodic, Flat, Flat))
    c = CenterField(grid)
    exact = XFaceField(grid)
    set!(c, x -> sin(x))
    set!(exact, x -> cos(x))
    fill_halo_regions!(c)
    error = Field(KernelFunctionOperation{Face, Center, Center}(fourth_order_∂xᵣᶠᶜᶜ, grid, c) - exact)
    return maximum(abs, error)
end

@testset "Fourth-order-accurate first derivative operators" begin
    @info "Testing fourth-order-accurate first derivative operators..."

    for arch in archs
        @testset "Convergence [$(typeof(arch))]" begin
            for ∂² in (∂x_∂x⁽⁴⁾, ∂y_∂y⁽⁴⁾, ∂z_∂z⁽⁴⁾)
                e₁, e₂ = second_derivative_errors(arch, ∂², 16), second_derivative_errors(arch, ∂², 32)
                @info "  Flux divergence of the fourth-order first derivative ($∂²): errors $e₁, $e₂, rate $(convergence_rate(e₁, e₂))"
                @test 3.9 < convergence_rate(e₁, e₂) < 4.1
            end

            # The derivative itself is the "flux form" (finite volume) face derivative,
            # which is second-order accurate pointwise
            e₁, e₂ = first_derivative_error(arch, 16), first_derivative_error(arch, 32)
            @test 1.9 < convergence_rate(e₁, e₂) < 2.1
        end

        @testset "Second-order fallback [$(typeof(arch))]" begin
            N = 16
            grid = RectilinearGrid(arch; size=(N, 4, N), x=(0, 1), y=(0, 1), z=(0, 1), halo=(3, 3, 3),
                                   topology=(Bounded, Periodic, Bounded))
            c = CenterField(grid)
            set!(c, (x, y, z) -> exp(x) * cos(3z))
            fill_halo_regions!(c)

            # Tracer fluxes on x-faces: the stencil i-2:i+1 includes a halo cell at i = 1, 2, N, N+1
            d = Field(KernelFunctionOperation{Face, Center, Center}(fourth_order_∂xᵣᶠᶜᶜ, grid, c) -
                      KernelFunctionOperation{Face, Center, Center}(∂xᵣᶠᶜᶜ, grid, c))
            @test maximum(abs, interior(d, 1:2, :, :)) == 0
            @test maximum(abs, interior(d, N:N+1, :, :)) == 0
            @test minimum(abs, interior(d, 3:N-1, :, :)) > 0

            # u-like fields differenced to centers: the stencil i-1:i+2 includes a halo face at i = 1, N
            u = XFaceField(grid)
            set!(u, (x, y, z) -> sin(π * x) * cos(3z))
            fill_halo_regions!(u)
            d = Field(KernelFunctionOperation{Center, Center, Center}(fourth_order_∂xᵣᶜᶜᶜ, grid, u) -
                      KernelFunctionOperation{Center, Center, Center}(∂xᵣᶜᶜᶜ, grid, u))
            @test maximum(abs, interior(d, 1, :, :)) == 0
            @test maximum(abs, interior(d, N, :, :)) == 0
            @test minimum(abs, interior(d, 2:N-1, :, :)) > 0

            # Immersed boundary: z-fluxes next to a flat bottom at z = 1/2 are second order
            ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(0.5))
            c = CenterField(ibg)
            set!(c, (x, y, z) -> exp(x) * cos(3z))
            fill_halo_regions!(c)
            d = Field(KernelFunctionOperation{Center, Center, Face}(fourth_order_∂zᶜᶜᶠ, ibg, c) -
                      KernelFunctionOperation{Center, Center, Face}(∂zᶜᶜᶠ, ibg, c))
            k₀ = N÷2 + 1 # the immersed boundary face
            @test maximum(abs, interior(d, :, :, k₀:k₀+1)) == 0
            @test minimum(abs, interior(d, :, :, k₀+2:N-1)) > 0

            # Grid traits
            @test fourth_order_x_stencil(grid) === Val(true)
            @test fourth_order_x_stencil(ibg) === Val(true)
            @test fourth_order_x_stencil(RectilinearGrid(size=4, x=[0, 1, 3, 4, 5], topology=(Bounded, Flat, Flat))) === Val(false)
            @test fourth_order_x_stencil(RectilinearGrid(size=4, z=(0, 1), topology=(Flat, Flat, Bounded))) === Val(false)
            @test fourth_order_x_stencil(LatitudeLongitudeGrid(size=(4, 4, 4), longitude=(0, 1), latitude=(0, 1), z=(0, 1))) === Val(false)
            partial = ImmersedBoundaryGrid(grid, PartialCellBottom(0.55))
            @test fourth_order_z_stencil(partial) === Val(false)
            @test fourth_order_x_stencil(partial) === Val(true)
        end

        @testset "Function arguments [$(typeof(arch))]" begin
            grid = RectilinearGrid(arch; size=8, x=(0, 1), halo=3, topology=(Periodic, Flat, Flat))
            c = CenterField(grid)
            set!(c, x -> x^3)
            fill_halo_regions!(c)
            @inline getc(i, j, k, grid, c) = @inbounds c[i, j, k]
            d = Field(KernelFunctionOperation{Face, Center, Center}(fourth_order_δxᶠᵃᵃ, grid, c) -
                      KernelFunctionOperation{Face, Center, Center}(fourth_order_δxᶠᵃᵃ, grid, getc, c))
            @test maximum(abs, d) == 0
        end
    end
end
