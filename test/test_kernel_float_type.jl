include("dependencies_for_runtests.jl")

using Adapt: adapt
using Oceananigans.Architectures: FloatTypeAdaptor
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

function has_float64(T, seen=Set{Any}())
    T in seen && return false
    push!(seen, T)
    T === Float64 && return true
    T isa UnionAll && return has_float64(T.body, seen)
    T isa Union && return has_float64(T.a, seen) || has_float64(T.b, seen)
    T isa DataType || return false
    return any(p -> p isa Type && has_float64(p, seen), T.parameters) ||
           any(t -> has_float64(t, seen), fieldtypes(T))
end

kernel_visible_components(model) =
    NamedTuple(name => getproperty(model, name)
               for name in propertynames(model)
               if name in (:grid, :clock, :velocities, :tracers, :advection, :closure, :closure_fields, :coriolis,
                           :buoyancy, :forcing, :free_surface, :stokes_drift, :background_fields))

field_boundary_conditions(model) = map(f -> f.boundary_conditions, merge(model.velocities, model.tracers))

function cpu_kernel_functions(mod, found=Set{Any}())
    for name in names(mod; all=true)
        isdefined(mod, name) || continue
        x = getfield(mod, name)
        if x isa Module && x !== mod && parentmodule(x) === mod
            cpu_kernel_functions(x, found)
        elseif x isa Function && startswith(string(name), "cpu_")
            push!(found, x)
        end
    end
    return found
end

cpu_kernel_specializations() =
    Set(mi for f in cpu_kernel_functions(Oceananigans) for m in methods(f) for mi in Base.specializations(m))

function float64_ssa_sites(mi)
    sites = Any[]
    for (code_info, _) in Base.code_typed_by_type(mi.specTypes; optimize=true)
        for (i, T) in enumerate(code_info.ssavaluetypes)
            T isa DataType || continue
            (T === Float64 || (T <: Tuple && Float64 in T.parameters)) && push!(sites, code_info.code[i])
        end
    end
    return sites
end

@testset "Float32 kernels" begin
    Oceananigans.defaults.FloatType = Float32
    try
        grid = RectilinearGrid(CPU(); size=(4, 4, 4), topology=(Periodic, Periodic, Bounded),
                               x=collect(range(0, 1000, length=5)), y=collect(range(0, 1000, length=5)), z=collect(range(-100, 0, length=5)))

        wind(x, y, t, p) = p.τ * sin(t / p.T)
        stokes_shear(z, t, p) = p.a * exp(z / p.d)
        stratification(x, y, z, t, p) = p.N² * z

        function models(FT)
            u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(wind, parameters=(τ=FT(1e-4), T=FT(1day))),
                                            bottom = BulkDrag(coefficient=FT(2.5e-3)))

            hydrostatic = HydrostaticFreeSurfaceModel(grid;
                                                      clock = Clock(time=zero(FT), last_Δt=FT(Inf)),
                                                      closure = CATKEVerticalDiffusivity(),
                                                      buoyancy = SeawaterBuoyancy(),
                                                      tracers = (:T, :S),
                                                      momentum_advection = WENO(),
                                                      tracer_advection = WENO(),
                                                      free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                                      coriolis = FPlane(latitude=45),
                                                      boundary_conditions = (; u=u_bcs),
                                                      forcing = (; T=Relaxation(rate=FT(1/10days), target=FT(20))))

            nonhydrostatic = NonhydrostaticModel(grid;
                                                 clock = Clock(time=zero(FT), last_Δt=FT(Inf)),
                                                 advection = WENO(),
                                                 closure = SmagorinskyLilly(),
                                                 buoyancy = BuoyancyTracer(),
                                                 tracers = :b,
                                                 coriolis = FPlane(f=1e-4),
                                                 stokes_drift = UniformStokesDrift(∂z_uˢ=stokes_shear, parameters=(a=FT(1e-2), d=FT(4))),
                                                 forcing = (; b=Relaxation(rate=FT(1/hour), mask=GaussianMask{:z}(center=FT(-100), width=FT(5)))),
                                                 background_fields = (; b=BackgroundField(stratification, parameters=(; N²=FT(1e-5)))))

            return (; hydrostatic, nonhydrostatic)
        end

        @info "  Testing that Float64 model components are converted to Float32 for kernels..."
        for model in models(Float64)
            components = merge(kernel_visible_components(model), (; boundary_conditions=field_boundary_conditions(model)))
            unconverted = [name for (name, c) in pairs(components) if has_float64(typeof(adapt(FloatTypeAdaptor{Float32}(), c)))]
            @test isempty(unconverted)
        end

        @info "  Testing that Float32 model kernels compute in Float32..."
        compiled_before = cpu_kernel_specializations()
        for model in models(Float32)
            run!(Simulation(model, Δt=60f0, stop_iteration=2, verbose=false))
        end
        leaks = Dict{Symbol, Vector{Any}}()
        for mi in setdiff(cpu_kernel_specializations(), compiled_before)
            sites = float64_ssa_sites(mi)
            isempty(sites) || append!(get!(leaks, mi.def.name, Any[]), sites)
        end
        @test isempty(leaks)
    finally
        Oceananigans.defaults.FloatType = Float64
    end
end
