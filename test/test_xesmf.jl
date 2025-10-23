include("dependencies_for_runtests.jl")

using XESMF
using SparseArrays
using LinearAlgebra

gaussian_bump(λ, φ; λ₀=0, φ₀=0, width=10) = exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2)

for arch in archs
    @testset "XESMF extension [$(typeof(arch))]" begin
        @info "Testing XESMF regridding [$(typeof(arch))]..."

        z = (-1, 0)
        southernmost_latitude = -80
        radius = Oceananigans.defaults.planet_radius

        llg_coarse = LatitudeLongitudeGrid(arch; z, radius,
                                           size = (176, 88, 1),
                                           longitude = (0, 360),
                                           latitude = (southernmost_latitude, 90))

        llg_fine = LatitudeLongitudeGrid(arch; z, radius,
                                         size = (360, 170, 1),
                                         longitude = (0, 360),
                                         latitude = (southernmost_latitude, 90))

        tg = TripolarGrid(arch; size=(360, 170, 1), z, southernmost_latitude, radius)


        for (src_grid, dst_grid) in ((llg_coarse, llg_fine),
                                     (llg_fine, llg_coarse),
                                     (tg, llg_fine))

            @info "  Regridding from $(nameof(typeof(src_grid))) to $(nameof(typeof(dst_grid)))"

            src_field = CenterField(src_grid)
            dst_field = CenterField(dst_grid)

            width = 12         # degrees
            set!(src_field,
                 (λ, φ, z) -> gaussian_bump(λ, φ; λ₀=150, φ₀=30, width) - 2gaussian_bump(λ, φ; λ₀=270, φ₀=-20, width))

            regridder = XESMF.Regridder(dst_field, src_field)

            if arch isa CPU
                @test regridder.weights isa SparseMatrixCSC
            elseif arch isa GPU{CUDABackend}
                @test regridder.weights isa CUDA.CUSPARSE.CuSparseMatrixCSC
            end

            if arch isa GPU
                cpu_regridder = on_architecture(CPU(), regridder)
                @test cpu_regridder.weights isa SparseMatrixCSC
                gpu_regridder = on_architecture(GPU(), cpu_regridder)
                @test gpu_regridder.weights isa CUDA.CUSPARSE.CuSparseMatrixCSC
            end

            regrid!(dst_field, regridder, src_field)

            # ∫ dst_field dA ≈ ∫ src_field dA
            @test @allowscalar isapprox(first(Field(Integral(dst_field, dims=(1, 2)))),
                                        first(Field(Integral(src_field, dims=(1, 2)))), rtol=1e-4)
        end
    end
end
