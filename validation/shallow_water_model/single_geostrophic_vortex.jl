
using Oceananigans
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation, shallow_water_velocities
using LinearAlgebra: norm
using NCDatasets, Printf, GLMakie
using Polynomials: fit
using Random

Nx, Ny = 100, 100

@inline gaussian(x, y, σ) = exp( - (x^2 + y^2)/(2*σ^2) )

# single vortex
function single_vortex_exp()
    
    grid = RectilinearGrid(GPU(); size = (Nx, Ny), x = (-0.5, 0.5), y = (-0.5, 0.5),
                           topology = (Periodic, Periodic, Flat))

    h₀ = -0.08
    H  = 1.0

    g  = 1.0
    f  = 10.0
    σ  = 0.1
    
    @inline bat(x, y, z) = 0.0

    @inline hᵢ(x, y, z) = H + h₀ * gaussian(x, y, σ)
    @inline uᵢ(x, y, z) = + g/f * ( - y / σ^2 * h₀ * gaussian(x, y, σ))
    @inline vᵢ(x, y, z) = - g/f * ( - x / σ^2 * h₀ * gaussian(x, y, σ))

    return grid, g, f, hᵢ, uᵢ, vᵢ, hᵢ, uᵢ, vᵢ, bat
end

# Lake at rest solution
function lake_at_rest()

    grid = RectilinearGrid(GPU(); size = (Nx, Ny), x = (-0.5, 0.5), y = (-0.5, 0.5),
                           topology = (Periodic, Periodic, Flat))

    g = 1.0
    f = 0.0
    σ = 0.1

    @inline bat(x, y, z) = 0.8 * gaussian(x, y, σ)

    @inline hᵢ(x, y, z) = 1 - bat(x, y, z)
    @inline uᵢ(x, y, z) = 0.0
    @inline vᵢ(x, y, z) = 0.0

    return grid, g, f, hᵢ, uᵢ, vᵢ, hᵢ, uᵢ, vᵢ, bat
end

grid, g, f, hᵢ, uᵢ, vᵢ, hₒ, uₒ, vₒ, bat = lake_at_rest()

@inline uhᵢ(x, y, z) = hᵢ(x, y, z) * uᵢ(x, y, z) 
@inline vhᵢ(x, y, z) = hᵢ(x, y, z) * vᵢ(x, y, z) 
@inline uhₒ(x, y, z) = hₒ(x, y, z) * uₒ(x, y, z) 
@inline vhₒ(x, y, z) = hₒ(x, y, z) * vₒ(x, y, z) 

coriolis = FPlane(; f)
solution  = [Dict(), Dict(), Dict(), Dict()]
model_sol = Dict() 

solh  = CenterField(grid)
solu  =  XFaceField(grid)
soluh =  XFaceField(grid)
set!(solh, hₒ)
set!(solu, uₒ)
set!(soluh, uhₒ)

solh  = Array(interior(solh,  :, :, 1))
solu  = Array(interior(solu,  :, :, 1))
soluh = Array(interior(soluh, :, :, 1))

function run_shallow_water_experiment(model, solution, form, order)
    if model.formulation isa ConservativeFormulation
        set!(model, h=hᵢ, uh=uhᵢ, vh=vhᵢ)
    else
        set!(model, h=hᵢ, u=uᵢ, v=vᵢ)
    end
    
    @info "starting time stepping"
    @show Δt = 5e-3
    for step in 1:600
        time_step!(model, Δt)
    end
    @info "finished time stepping"
    
    u₁, u₂, h = model.solution

    solu₁ = model.formulation isa ConservativeFormulation ? soluh : solu

    solution[1][(form, order)] = norm((Array(interior(h,  :, :, 1)) .- solh) , 2) ./ (Nx*Ny)^(1/2) 
    solution[2][(form, order)] = norm((Array(interior(u₁, :, :, 1)) .- solu₁), 2) ./ (Nx*Ny)^(1/2) 
    solution[3][(form, order)] = maximum(abs, (Array(interior(h,  :, :, 1)) .- solh)) 
    solution[4][(form, order)] = maximum(abs, (Array(interior(u₁, :, :, 1)) .- solu₁)) 
end

@inline weno_advection(::Val{:conservative}, order)     = WENO(; order)
@inline weno_advection(::Val{:vorticitystencil}, order) = WENO(; order, vector_invariant = VorticityStencil())
@inline weno_advection(::Val{:velocitystencil}, order)  = WENO(; order, vector_invariant = VelocityStencil())

@inline Formulation(::Val{:conservative})     = ConservativeFormulation()
@inline Formulation(::Val{:vorticitystencil}) = VectorInvariantFormulation()
@inline Formulation(::Val{:velocitystencil})  = VectorInvariantFormulation()

for form in [:conservative, :vorticitystencil, :velocitystencil]
    for order in [3, 5, 7, 9]

        @info "starting with $form and order $order"

        adv = weno_advection(Val(form), order)
        model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration = g,
                                    mass_advection = WENO(; order),
                                    momentum_advection = adv,
                                    bathymetry = bat,
                                    formulation = Formulation(Val(form)))

        run_shallow_water_experiment(model, solution, form, order)

        model_sol[(form, order)] = model
    end
end

using PrettyTables

cases = [:Conservative, :VI_Vorticity, :VI_Velocity,
         :Conservative, :VI_Vorticity, :VI_Velocity,
         :Conservative, :VI_Vorticity, :VI_Velocity,
         :Conservative, :VI_Vorticity, :VI_Velocity]

order = [3, 3, 3,
         5, 5, 5,
         7, 7, 7,
         9, 9, 9]

herr₂ = []
uerr₂ = []
herr∞ = []
uerr∞ = []

for order in [3, 5, 7, 9]
    for form in [:conservative, :vorticitystencil, :velocitystencil]
        push!(herr₂, solution[1][(form, order)])  
        push!(uerr₂, solution[2][(form, order)])    
        push!(herr∞, solution[3][(form, order)])  
        push!(uerr∞, solution[4][(form, order)])   
    end
end

data = hcat(cases, order, herr₂, uerr₂, herr∞, uerr∞)

header = (["Formulation", "Order", "L₂ error(h)", "L₂ error(u or uh)", "L∞ error(h)", "L∞ error(u or uh)"])
pretty_table(data; header, tf = tf_unicode_rounded, header_crayon = crayon"yellow bold", formatters = ft_printf("%.2e", 3:6))

ordr = [Dict(), Dict()]
for order in [3, 5, 7, 9]
    for form in [:conservative, :vorticitystencil, :velocitystencil]
        temph = zeros(5)
        tempu = zeros(5)
        for (idx, sol, prev) in zip([1, 2, 3, 4], [sol64, sol128, sol256, sol512], [sol32, sol64, sol128, sol256])
            temph[idx] = log(prev[1][(form, order)] / sol[1][(form, order)]) / log(2)
            tempu[idx] = log(prev[2][(form, order)] / sol[2][(form, order)]) / log(2)
        end
        ordr[1][(form, order)] = temph
        ordr[2][(form, order)] = tempu
    end
end 
