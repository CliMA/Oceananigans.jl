import Oceananigans.Grids: short_show

#####
##### BoundaryCondition
#####

print_condition(n::Union{Nothing, Number}) = "$n"
print_condition(A::AbstractArray) = "$(Base.dims2string(size(A))) $(typeof(A))"
print_condition(bf::Union{ParameterizedDiscreteBoundaryFunction, BoundaryFunction}) = print_condition(bf.func)

function print_condition(f::Function)
    ms = methods(f).ms
    length(ms) == 1 && return "$(ms[1])"
    return "$(ms)"
end

Base.show(io::IO, bc::BC{C, T}) where {C, T} =
    print(io, "BoundaryCondition: type=$C, condition=$(print_condition(bc.condition))")

#####
##### FieldBoundaryConditions
#####

bctype_str(::FBC)  = "Flux"
bctype_str(::PBC)  = "Periodic"
bctype_str(::NFBC) = "NormalFlow"
bctype_str(::VBC)  = "Value"
bctype_str(::GBC)  = "Gradient"
bctype_str(::ZFBC) = "ZeroFlux"

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
