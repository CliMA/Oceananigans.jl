import Base: show

const DFBC = DefaultPrognosticFieldBoundaryCondition

bc_str(::FBC)     = "Flux    "
bc_str(::PBC)     = "Periodic"
bc_str(::OBC)     = "Open    "
bc_str(::VBC)     = "Value   "
bc_str(::GBC)     = "Gradient"
bc_str(::ZFBC)    = "ZeroFlux"
bc_str(::DFBC)    = "Default "
bc_str(::Nothing) = "Nothing "

#####
##### BoundaryCondition
#####

condition_str(n::Union{Nothing, Number}) = "$n"
condition_str(A::AbstractArray) = "$(Base.dims2string(size(A))) $(typeof(A))"
condition_str(bf::Union{DiscreteBoundaryFunction, ContinuousBoundaryFunction}) = condition_str(bf.func)

function condition_str(f::Function)
    ms = methods(f).ms
    length(ms) == 1 && return "$(ms[1])"
    return "$(ms)"
end

show(io::IO, bc::BoundaryCondition) =
    print(io, "BoundaryCondition: classification=", rstrip(bc_str(bc)), ", condition=", condition_str(bc.condition))

#####
##### FieldBoundaryConditions
#####

Base.summary(fbcs::FieldBoundaryConditions) = "FieldBoundaryConditions"
    
show_field_boundary_conditions(bcs::FieldBoundaryConditions, padding="") =
    string("Oceananigans.FieldBoundaryConditions, with boundary conditions", '\n',
           padding, "├── west: ",     typeof(bcs.west), '\n',
           padding, "├── east: ",     typeof(bcs.east), '\n',
           padding, "├── south: ",    typeof(bcs.south), '\n',
           padding, "├── north: ",    typeof(bcs.north), '\n',
           padding, "├── bottom: ",   typeof(bcs.bottom), '\n',
           padding, "├── top: ",      typeof(bcs.top), '\n',
           padding, "└── immersed: ", typeof(bcs.immersed))

Base.show(io::IO, fieldbcs::FieldBoundaryConditions) = print(io, show_field_boundary_conditions(fieldbcs))
