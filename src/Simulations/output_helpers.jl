#####
##### Lend a helping hand
#####

struct GenericJLD2Name end
unique_name(::GenericJLD2Name, existing) = unique_name(:jld2, existing)

struct JLD2Format end

"""
    output!(simulation, outputs [, format=JLD2Format()]; kw...)

"""
output!(simulation, outputs; kw...) = output!(simulation, outputs, JLD2Format(); kw...)

function output!(simulation, outputs, ::JLD2Format; kw...)
    if !(name ∈ keys(kw))
        name = GenericJLD2Name()        
    end
    name = unique_name(name, keys(simulation.output_writers))
    ow = JLD2OutputWriter(simulation.model, outputs; kw...)
    simulation.output_writers[name] = ow
    return nothing
end

