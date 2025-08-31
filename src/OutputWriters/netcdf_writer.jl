#####
##### NetCDFWriter struct definition
#####
##### NetCDFWriter functionality is implemented in ext/OceananigansNCDatasetsExt
#####

mutable struct NetCDFWriter{G, D, O, T, A, FS, DN} <: AbstractOutputWriter
    grid :: G
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    global_attributes :: Dict
    output_attributes :: Dict
    dimensions :: Dict
    with_halos :: Bool
    include_grid_metrics :: Bool
    overwrite_existing :: Bool
    verbose :: Bool
    deflatelevel :: Int
    part :: Int
    file_splitting :: FS
    dimension_name_generator :: DN
end

function NetCDFWriter(model, outputs; kw...)
    @warn "`using NCDatasets` is required (without erroring!) to use `NetCDFWriter`."
    throw(MethodError(NetCDFWriter, (model, outputs)))
end
