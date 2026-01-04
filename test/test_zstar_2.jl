using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: KernelParameters, launch!
using Statistics

Oceananigans.defaults.FloatType = BigFloat

arch = CPU()
z_stretched = MutableVerticalDiscretization(collect(-20:0))
grid = TripolarGrid(arch; size = (30, 30, 20), z = z_stretched)

# Code credit:
# https://github.com/PRONTOLab/GB-25/blob/682106b8487f94da24a64d93e86d34d560f33ffc/src/model_utils.jl#L65
function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end

zb = - 20
h  = - zb + 10
gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))
free_surface = SplitExplicitFreeSurface(grid; substeps=20)

model = HydrostaticFreeSurfaceModel(; grid,
                                      free_surface,
                                      tracers = (:b, :c, :constant),
                                      buoyancy = BuoyancyTracer(),
                                      timestepper = :SplitRungeKutta3,
                                      vertical_coordinate = ZStarCoordinate())

bᵢ(x, y, z) = y > 0 ? 0.06 : 0.01

# Instead of initializing with random velocities, infer them from a random initial streamfunction
# to ensure the velocity field is divergence-free at initialization.
ψ = Field{Face, Face, Center}(grid)

mean_xspacing = mean(xspacings(grid, Face(), Face(), Center()))
mean_yspacing = mean(yspacings(grid, Face(), Face(), Center()))
Δ = mean((mean_xspacing, mean_yspacing))
U = 1

# Set streamfunction amplitude to Δ * U to yield velocities of order U.
set!(ψ, U * Δ * rand(size(ψ)...))
Oceananigans.BoundaryConditions.fill_halo_regions!(ψ)

uᵢ = - ∂y(ψ)
vᵢ = ∂x(ψ)

set!(model, c = (x, y, z) -> rand(), u = uᵢ, v = vᵢ, b = bᵢ, constant = 1)

Δt = 2minutes

using KernelAbstractions: @kernel, @index

@kernel function _subtract_line!(∫f, grid, f)
    i, k = @index(Global, NTuple)
    j = size(grid, 2)

    immersed = Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, grid)

    if !immersed
        ∫f[] -= f[i, j, k] 
    end
end

function my_average(field)
    f_V = field * Oceananigans.Operators.volume
    ∫f_V = Ref(sum(f_V))
    V = KernelFunctionOperation{Oceananigans.Fields.location(field)...}(Oceananigans.Operators.Vᶜᶜᶜ, field.grid)
    ∫V = Ref(sum(V))
    
    Nx, Ny, Nz = size(field.grid)
    params = KernelParameters(Nx÷2+1:Nx, 1:Nz)
    
    launch!(CPU(), field.grid, params, _subtract_line!, ∫f_V, field.grid, f_V)
    launch!(CPU(), field.grid, params, _subtract_line!, ∫V,   field.grid, V)
    
    return ∫f_V[] / ∫V[]
end 

my_average(field) = Field(Average(field))[1, 1, 1]

∫b1 = my_average(deepcopy(model.tracers.b))
∫c1 = my_average(deepcopy(model.tracers.c))
compute!(∫b1)
compute!(∫c1)

w  = model.velocities.w
Nz = model.grid.Nz

FT = Oceananigans.defaults.FloatType

bc = FT[]
cc = FT[]
wc = FT[]
constmx = FT[]
constmn = FT[]

for step in 1:300
    time_step!(model, Δt)

    ∫b = my_average(model.tracers.b)
    ∫c = my_average(model.tracers.c)

    condition = ∫b ≈ ∫b1
    if !condition
        @info "Stopping early: buoyancy not conserved at step $step: $((∫b - ∫b1) / ∫b1)"
    end
    # @test condition

    condition = ∫c ≈ ∫c1
    if !condition
        @info "Stopping early: c tracer not conserved at step $step: $((∫c - ∫c1) / ∫c1)"
    end
    # @test condition

    condition = maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
    if !condition
        @info "Stopping early: nonzero vertical velocity at top at step $step: $(maximum(abs, interior(w, :, :, Nz+1)))"
    end
    
    push!(bc, (∫b - ∫b1) / ∫b1)
    push!(cc, (∫c - ∫c1) / ∫c1)
    push!(wc, maximum(abs, interior(w, :, :, Nz+1)))
    push!(constmx, maximum(model.tracers.constant))
    push!(constmn, minimum(model.tracers.constant))
    
    # Constancy preservation test
    # if test_local_conservation
        @test maximum(model.tracers.constant) ≈ 1
        @test minimum(model.tracers.constant) ≈ 1
    # end
end