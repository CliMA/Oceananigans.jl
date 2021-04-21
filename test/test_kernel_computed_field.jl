using Oceananigans.Fields: KernelComputedField
using KernelAbstractions: @kernel, @index

grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1),
                            topology=(Periodic, Periodic, Bounded))

@kernel function double!(doubled_field, grid, single_field)
    i, j, k = @index(Global, NTuple)
    @inbounds doubled_field[i, j, k] = 2 * single_field[i, j, k]
end

@kernel function multiply!(multiplied_field, grid, field, multiple)
    i, j, k = @index(Global, NTuple)
    @inbounds multiplied_field[i, j, k] = multiple * field[i, j, k]
end

for arch in archs
    @testset "KernelComputedField [$(typeof(arch))]" begin
        @info "  Testing KernelComputedField..."

        single_field = Field(Center, Center, Center, arch, grid)

        doubled_field = KernelComputedField(Center, Center, Center,
                                            double!, arch, grid,
                                            computed_dependencies = single_field)

        # Test that the constructor worked
        @test doubled_field isa KernelComputedField

        multiple = 3
        multiplied_field = KernelComputedField(Center, Center, Center,
                                               multiply!, arch, grid,
                                               computed_dependencies = doubled_field,
                                               parameters = multiple)

        # Test that the constructor worked
        @test multiplied_field isa KernelComputedField

        set!(single_field, π)
        @test single_field[1, 1, 1] == convert(eltype(single_field), π)

        compute!(doubled_field)
        @test doubled_field[1, 1, 1] == 2π

        # Test boundary conditions / halo filling
        @test doubled_field[0, 1, 1] == doubled_field[2, 1, 1] # periodic
        @test doubled_field[1, 0, 1] == doubled_field[1, 2, 1] # periodic
        @test doubled_field[1, 1, 0] == doubled_field[1, 1, 1] # no flux
        @test doubled_field[1, 1, 1] == doubled_field[1, 1, 2] # no flux

        set!(doubled_field, 0)
        compute!(multiplied_field)

        @test doubled_field[1, 1, 1] == 2π
        @test multiplied_field[1, 1, 1] == multiple * 2 * π

        doubled_face_field = KernelComputedField(Center, Center, Face,
                                                 double!, arch, grid,
                                                 computed_dependencies = single_field)

        # Test that nothing happens for fields on faces in bounded directions
        compute!(doubled_face_field)
        @test doubled_face_field[1, 1, 0] == 0
        @test doubled_face_field[1, 1, 3] == 0
    end
end
