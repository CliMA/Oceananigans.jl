#####
##### Lend a helping hand
#####

struct GenericJLD2Name end
unique_name(::GenericJLD2Name, existing) = unique_name(:jld2, existing)

struct GenericNetCDFName end
unique_name(::GenericNetCDFName, existing) = unique_name(:nc, existing)

struct GenericPickUpName end
unique_name(::GenericPickUpName, existing) = unique_name(:pickup, existing)

struct JLD2Format end
struct NetCDFFormat end
struct Checkpoint end

"""
    output!(simulation, outputs [, format=JLD2Format()]; kw...)

"""
output!(simulation, outputs; kw...) = output!(simulation, outputs, JLD2Format(); kw...)

function output!(simulation, outputs, ::JLD2Format; kw...)
    if !("name" ∈ keys(kw))
        name = GenericJLD2Name()
    end
    name = unique_name(name, keys(simulation.output_writers))
    ow = JLD2OutputWriter(simulation.model, outputs; kw...)
    simulation.output_writers[name] = ow
    return nothing
end

function output!(simulation, outputs, ::NetCDFFormat; kw...)
    if !("name" ∈ keys(kw))
        name = GenericNetCDFName()        
    end
    name = unique_name(name, keys(simulation.output_writers))
    ow = NetCDFOutputWriter(simulation.model, outputs; kw...)
    simulation.output_writers[name] = ow
    return nothing
end

function output!(simulation, outputs, ::Checkpoint; kw...)
    if !(name ∈ keys(kw))
        name = GenericPickUpName()        
    end
    name = unique_name(name, keys(simulation.output_writers))
    ow = Checkpointer(simulation.model, kw...)
    simulation.output_writers[name] = ow
    return nothing
end
