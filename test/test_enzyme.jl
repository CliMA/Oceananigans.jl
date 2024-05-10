using Oceananigans
using Enzyme

# Required presently
Enzyme.API.runtimeActivity!(true)

EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true

f(grid) = CenterField(grid)

@testset "Enzyme Unit Tests" begin
    arch=CPU()
    FT=Float64

    N = 100
    topo = (Periodic, Flat, Flat)
    grid = RectilinearGrid(arch, FT, topology=topo, size=N, halo=2, x=(-1, 1), y=(-1, 1), z=(-1, 1))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, typeof(Const(grid)))

    tape, primal, shadow = fwd(Const(f), Const(grid) )

    @show tape, primal, shadow

    @test size(primal) == size(shadow)
end

@testset "Test Constructor Any Bug" begin
    using Oceananigans.Fields: FunctionField
    using Oceananigans: architecture
    using KernelAbstractions

    Enzyme.API.runtimeActivity!(true)
    Enzyme.API.looseTypeAnalysis!(true)
    Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
    Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
    Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

    Nx = Ny = 64
    Nz = 8

    x = y = (-π, π)
    z = (-0.5, 0.5)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracers = :c,
                                        buoyancy = nothing)

    function set_initial_condition!(model_tracer, amplitude)
        # Set initial condition
        amplitude = Ref(amplitude)

        # This has a "width" of 0.1
        cᵢ(x, y, z) = amplitude[]
        temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

        temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
        grid = model_tracer.grid
        arch = architecture(model_tracer)
        bc′ = temp

        param = Oceananigans.Utils.KernelParameters(size(model_tracer), map(offset_index, model_tracer.indices))
        Oceananigans.Utils.launch!(arch, grid, param, _broadcast_kernel!, model_tracer, bc′)

        return nothing
    end

    @inline offset_index(::Colon) = 0
    @inline offset_index(range::UnitRange) = range[1] - 1

    @kernel function _broadcast_kernel!(dest, bc)
    i, j, k = @index(Global, NTuple)
    @inbounds dest[i, j, k] = bc[i, j, k]
    end

    model_tracer = model.tracers.c
    dmodel_tracer = Enzyme.make_zero(model_tracer)

    amplitude = 1.0
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
    temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

    temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
    grid = model_tracer.grid
    arch = architecture(model_tracer)
    bc′ = temp

    param = Oceananigans.Utils.KernelParameters(size(model_tracer), map(offset_index, model_tracer.indices))
    Oceananigans.Utils.launch!(arch, grid, param, _broadcast_kernel!, model_tracer, bc′)

    try
        autodiff(Enzyme.Reverse,
                Oceananigans.Utils.launch!,
                Const(arch),
                Const(grid),
                Const(param),
                Const(_broadcast_kernel!),
                Duplicated(model_tracer, dmodel_tracer),
                Const(bc′))
    catch
        @show "Constructor for type 'Any' bug in Enzyme - it is likely something in Julia or KernelAbstractions.jl is causing broadcasted arrays in Oceananigans to break with AD."

end
