import Base: show
import Oceananigans: short_show

bcclassification_str(::FBC)  = "Flux"
bcclassification_str(::PBC)  = "Periodic"
bcclassification_str(::OBC)  = "Open"
bcclassification_str(::VBC)  = "Value"
bcclassification_str(::GBC)  = "Gradient"
bcclassification_str(::ZFBC) = "ZeroFlux"
bcclassification_str(::Nothing) = "Nothing"

#####
##### BoundaryCondition
#####

print_condition(n::Union{Nothing, Number}) = "$n"
print_condition(A::AbstractArray) = "$(Base.dims2string(size(A))) $(typeof(A))"
print_condition(bf::Union{DiscreteBoundaryFunction, ContinuousBoundaryFunction}) = print_condition(bf.func)

function print_condition(f::Function)
    ms = methods(f).ms
    length(ms) == 1 && return "$(ms[1])"
    return "$(ms)"
end

show(io::IO, bc::BoundaryCondition) =
    print(io, "BoundaryCondition: type=$(bcclassification_str(bc)), condition=$(print_condition(bc.condition))")

#####
##### FieldBoundaryConditions
#####

short_show(fbcs::FieldBoundaryConditions) =
    string("x=(west=$(bcclassification_str(fbcs.x.left)), east=$(bcclassification_str(fbcs.x.right))), ",
           "y=(south=$(bcclassification_str(fbcs.y.left)), north=$(bcclassification_str(fbcs.y.right))), ",
           "z=(bottom=$(bcclassification_str(fbcs.z.left)), top=$(bcclassification_str(fbcs.z.right)))")

show_field_boundary_conditions(bcs::FieldBoundaryConditions, padding="") =
    string("Oceananigans.FieldBoundaryConditions (NamedTuple{(:x, :y, :z)}), with boundary conditions", '\n',
           padding, "├── x: ", typeof(bcs.x), '\n',
           padding, "├── y: ", typeof(bcs.y), '\n',
           padding, "└── z: ", typeof(bcs.z))

Base.show(io::IO, fieldbcs::FieldBoundaryConditions) = print(io, show_field_boundary_conditions(fieldbcs))
