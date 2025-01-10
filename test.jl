using Oceananigans

grid = RectilinearGrid(GPU(), 
                       size=(1, 1, 1), 
                       extent=(1000, 1000, 1000), 
                       topology=(Bounded, Bounded, Bounded))

@inline field_dependent_fun(ξ, η, t, u, v) = - sqrt(u^2 + v^2)

T_east_bcs = ValueBoundaryCondition(field_dependent_fun, field_dependencies=(:u, :v))
T_bcs = FieldBoundaryConditions(east=T_east_bcs)

model = HydrostaticFreeSurfaceModel(; grid, boundary_conditions=(; T=T_bcs), tracers = :T)