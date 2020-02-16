import Oceananigans.Grids: short_show

#####
##### BoundaryCondition
#####

Base.show(io::IO, bc::BC{C, T}) where {C, T} =
    println(io, "BoundaryCondition: type=$C, condition=$(bc.condition)")

#####
##### FieldBoundaryConditions
#####

bctype_str(::FBC)  = "Flux"
bctype_str(::PBC)  = "Periodic"
bctype_str(::NPBC) = "NoPenetration"
bctype_str(::VBC)  = "Value"
bctype_str(::GBC)  = "Gradient"
bctype_str(::NFBC) = "NoFlux"

short_show(fbcs::FieldBoundaryConditions) =
    string("x=(west=$(bctype_str(fbcs.x.left)), east=$(bctype_str(fbcs.x.right))), ",
           "y=(south=$(bctype_str(fbcs.y.left)), north=$(bctype_str(fbcs.y.right))), ",
           "z=(bottom=$(bctype_str(fbcs.z.left)), top=$(bctype_str(fbcs.z.right)))")

show_field_boundary_conditions(bcs::FieldBoundaryConditions, padding="") =
    string("Oceananigans.FieldBoundaryConditions (NamedTuple{(:x, :y, :z)}), with boundary conditions", '\n',
           padding, "├── x: ", typeof(bcs.x), '\n',
           padding, "├── y: ", typeof(bcs.y), '\n',
           padding, "└── z: ", typeof(bcs.z))

Base.show(io::IO, fieldbcs::FieldBoundaryConditions) = print(io, show_field_boundary_conditions(fieldbcs))

#####
##### ModelBoundaryConditions
#####

function show_solution_boundary_conditions(bcs, padding)
    stringtuple = Tuple(string(
                  padding, "├── ", field, ": ",
                  show_field_boundary_conditions(getproperty(bcs, field), padding * "│   "), '\n')
                  for field in propertynames(bcs)[1:end-1])
    return string("Oceananigans.SolutionBoundaryConditions ",
                  "(NamedTuple{(:u, :v, :w, ...)}) with boundary conditions ", '\n', stringtuple...,
                  padding, "└── ", propertynames(bcs)[end], ": ",
                  show_field_boundary_conditions(bcs[end], padding * "    "))
end

Base.show(io::IO, bcs::ModelBoundaryConditions) =
    print(io,
          "Oceananigans.ModelBoundaryConditions (NamedTuple{(:solution, :tendency, :pressure, :diffusivities)}) with ", '\n',
          "├── solution: ", show_solution_boundary_conditions(bcs.solution, "│   "), '\n',
          "├── tendency: ", show_solution_boundary_conditions(bcs.tendency, "│   "), '\n',
          "└── pressure: ", show_field_boundary_conditions(bcs.pressure, "    "))
