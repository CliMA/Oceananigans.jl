import Logging, Dates
export ModelLogger, shouldlog, min_enabled_level, catch_exceptions, handle_message


# ------------------------------------------------------------------------------------
# ModelLogger
_model_logger_docs = """

    ModelLogger(stream::IO, level::LogLevel)

Based on Logging.SimpleLogger it tries to log all messages in the following format

    [dd/mm/yyyy HH:MM:SS] module source_file:line_number: message

The logger will handle any message from @debug up.
"""
struct ModelLogger <: Logging.AbstractLogger
    stream::IO
    min_level::Logging.LogLevel
    message_limits::Dict{Any,Int}
end
ModelLogger(stream::IO=stderr, level=Diagnostic) = ModelLogger(stream, level, Dict{Any,Int}())

Logging.shouldlog(logger::ModelLogger, level, _module, group, id) = get(logger.message_limits, id, 1) > 0

Logging.min_enabled_level(logger::ModelLogger) = logger.min_level

Logging.catch_exceptions(logger::ModelLogger) = false

function Logging.handle_message(logger::ModelLogger, level, message, _module, group, id, filepath, line; maxlog = nothing, kwargs...)
    if maxlog !== nothing && maxlog isa Integer
        remaining = get!(logger.message_limits, id, maxlog)
        logger.message_limits[id] = remaining - 1
        remaining > 0 || return
    end
    buf = IOBuffer()
    iob = IOContext(buf, logger.stream)
    level = level == Logging.Warn ? "Warning" : string(level)
    module_name = something(_module, "nothing")
    file_name = something(filepath, "nothing")
    line_number = something(line, "nothing")
    msg_timestamp = Dates.format(Dates.now(), "[dd/mm/yyyy HH:MM:SS]")
    formatted_message = "$msg_timestamp $module_name $file_name.jl:$line_number: $message"
    println(iob, formatted_message)
    write(logger.stream, take!(buf))
    nothing
end

# -------------------------------------------------------------------------------------
# Custom LogStates

const Diagnostic = Logging.LogLevel(-500)  # Sits between Debug and Info

# -------------------------------------------------------------------------------------
# Custom Logging Macros

# macro diagnostic(exs...) Logging.logmsg_code((@Logging._sourceinfo)..., :Diagnostic, exs...) end 