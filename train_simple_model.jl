using JLD2, FileIO
using ArgParse
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets
using BSON: @save


@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 1    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
    data::String = "/nfs/raid66/u11/users/brozonoy-ad/Oceananigans.jl/tdata.jld2"      ## path to jld2 data file
end


function getdata(args)

    data = load(args.data)

    X, y = [], []
    for (timestep, timestep_data) in data
        push!(X, timestep_data[timestep]["rhs"])  # shape 3600
        push!(y, reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,)))    # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
    end
    ## Create DataLoader object (mini-batch iterator)
    train_loader = DataLoader((X, y), batchsize=args.batchsize, shuffle=true)

    return train_loader
end


function MLP(; inputsize=(3600,1))
    return Chain( Dense(prod(inputsize), 256, relu),
                Dense(256, 62*62))
end


# function LeNet5(; imgsize=(62,62,1), nclasses=) 
#     out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
#     return Chain(
#             Conv((5, 5), imgsize[end]=>6, relu),
#             MaxPool((2, 2)),
#             Conv((5, 5), 6=>16, relu),
#             MaxPool((2, 2)),
#             flatten,
#             Dense(prod(out_conv_size), 120, relu), 
#             Dense(120, 84, relu), 
#             Dense(84, nclasses)
#           )
# end


function loss(data_loader, model, device)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x[1]), device(y[1])
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        num +=  size(x)[end]
    end
    return ls / num
end


function train(; kws...)

    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Create test and train dataloaders
    train_loader = getdata(args)

    ## Construct model
    model = MLP() |> device
    ps = Flux.params(model) ## model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)
    
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x[1]), device(y[1]) ## transfer data to device
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) ## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        
        ## Report on train and test
        train_loss = loss(train_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss")
    end

    return model
end


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--data", "-i"
            help = "path to jld2 in jld2 format"
            arg_type = String
            required = true
        "--model", "-o"
            help = "path to save model"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__

    parsed_args = parse_commandline()
    trained_model = train(; data=parsed_args["data"])
    @save parsed_args["model"] trained_model

end
