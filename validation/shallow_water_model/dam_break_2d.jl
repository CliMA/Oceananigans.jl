using Oceananigans
using Oceananigans.Grids: xnode, ynode
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation, shallow_water_velocities
using LinearAlgebra: norm
using NCDatasets, Printf, GLMakie
using Polynomials: fit
using Random

Nx, Ny = 200, 200

grid = RectilinearGrid(CPU(); size = (Nx, Ny), x = (-2, 2), y = (-2, 2),
                        topology = (Bounded, Bounded, Flat))
    
g = 10.0

hᵢ(x, y, z) = y <= x ? 1.0 : 0.0
uᵢ(x, y, z) = 0.0
vᵢ(x, y, z) = 0.0

uhᵢ(x, y, z) = hᵢ(x, y, z) * uᵢ(x, y, z) 
vhᵢ(x, y, z) = hᵢ(x, y, z) * vᵢ(x, y, z) 

final_time = 1.0

solution  = [Dict(), Dict(), Dict(), Dict()]
model_sol = Dict() 

solh  = CenterField(grid)
solu  =  XFaceField(grid)
soluh =  XFaceField(grid)

function exact_solution_h(x_old, y_old, z; t = final_time)
    x = x_old + y_old * √2
    y = y_old - x_old * √2

    if y < - 1t
        return 1.0
    elseif y < 2t
        return 1/9*(2 - y/t)^2
    else
        0.0
    end
end

function exact_solution_u(x_old, y_old, z; t = final_time)
    x = x_old + y_old * √2
    y = y_old - x_old * √2

    if x < - 1t
        u = 0
    elseif x < 2t
        u = 2/3*(2 - x/t)
    else
        u = 0
    end

    return u * cosd(45)
end

exact_solution_uh(x, y, z) = exact_solution_h(x, y, z) * exact_solution_u(x, y, z)

set!(solh,  exact_solution_h)
set!(solu,  exact_solution_u)
set!(soluh, exact_solution_uh)

solh  = Array(interior(solh,  :, :, 1))
solu  = Array(interior(solu,  :, :, 1))

function run_shallow_water_experiment(model, solution, form, order)
    if model.formulation isa ConservativeFormulation
        set!(model, h=hᵢ, uh=uhᵢ, vh=vhᵢ)
    else
        set!(model, h=hᵢ, u=uᵢ, v=vᵢ)
    end
    
    @info "starting time stepping"
    Δt = final_time / 1000
    for step in 1:1000
        time_step!(model, Δt)
    end
    @info "finished time stepping"
    
    u₁, u₂, h = model.solution

    solu₁ = model.formulation isa ConservativeFormulation ? soluh : solu

    nx = model.grid.Nx ÷ 4
    ny = model.grid.Ny ÷ 4

    mx = Int(nx * 3)
    my = Int(nx * 3)

    solution[1][(form, order)] = norm((Array(interior(h, nx:mx, ny:my, 1)) .- solh[nx:mx, ny:my]) , 2) ./ (Nx*Ny)^(1/2) 
    solution[2][(form, order)] = norm((Array(interior(u₁, nx:mx, ny:my, 1)) .- solu₁[nx:mx, ny:my]), 2) ./ (Nx*Ny)^(1/2) 
    solution[3][(form, order)] = maximum(abs, (Array(interior(h, nx:mx, ny:my, 1)) .- solh[nx:mx, ny:my])) 
    solution[4][(form, order)] = maximum(abs, (Array(interior(u₁, nx:mx, ny:my, 1)) .- solu₁[nx:mx, ny:my])) 
end

@inline weno_advection(::Val{:conservative}, order)     = WENO(; order)
@inline weno_advection(::Val{:vorticitystencil}, order) = WENO(; order, vector_invariant = VorticityStencil())
@inline weno_advection(::Val{:velocitystencil}, order)  = WENO(; order, vector_invariant = VelocityStencil())

@inline Formulation(::Val{:conservative})     = ConservativeFormulation()
@inline Formulation(::Val{:vorticitystencil}) = VectorInvariantFormulation()
@inline Formulation(::Val{:velocitystencil})  = VectorInvariantFormulation()

for form in [:vorticitystencil, :velocitystencil]
    for order in [3, 5]    
        @info "starting with $form and order $order"

        adv = weno_advection(Val(form), order)
        model = ShallowWaterModel(; grid, coriolis=nothing, gravitational_acceleration = g,
                                    mass_advection = WENO(; order),
                                    momentum_advection = adv,
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
