import Oceananigans.Grids: short_show

#####
##### BoundaryCondition
#####

Base.show(io::IO, bc::BC{C, T}) where {C, T} =
    print(io, "BoundaryCondition: type=$C, condition=$(bc.condition)")

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
