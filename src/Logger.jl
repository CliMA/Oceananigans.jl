module Logger

export ModelLogger, Diagnostic, Setup

using Dates
using Logging
using Crayons

const RED    = Crayon(foreground=:red)
const YELLOW = Crayon(foreground=:light_yellow)
const CYAN   = Crayon(foreground=:cyan)
const BLUE   = Crayon(foreground=:blue)

const BOLD      = Crayon(bold=true)
const UNDERLINE = Crayon(underline=true)

"""
    ModelLogger(stream::IO, level::LogLevel)

Based on Logging.SimpleLogger, it tries to log all messages in the following format:

[yyyy/mm/dd HH:MM:SS.sss] log_level message @ source_file:line_number
"""
struct ModelLogger <: Logging.AbstractLogger
            stream :: IO
         min_level :: Logging.LogLevel
    message_limits :: Dict{Any,Int}
end

ModelLogger(stream::IO=stdout, level=Logging.Info) = ModelLogger(stream, level, Dict{Any,Int}())

Logging.shouldlog(logger::ModelLogger, level, _module, group, id) = get(logger.message_limits, id, 1) > 0

Logging.min_enabled_level(logger::ModelLogger) = logger.min_level

Logging.catch_exceptions(logger::ModelLogger) = false

function level_to_string(level)
    level == Logging.Error && return "ERROR"
    level == Logging.Warn  && return "WARN "
    level == Logging.Info  && return "INFO "
    level == Logging.Debug && return "DEBUG"
    return string(level)
end

function level_to_crayon(level)
    level == Logging.Error && return RED
    level == Logging.Warn  && return YELLOW
    level == Logging.Info  && return CYAN
    level == Logging.Debug && return BLUE
    return identity
end

function Logging.handle_message(logger::ModelLogger, level, message, _module, group, id,
                                filepath, line; maxlog = nothing, kwargs...)

    if !isnothing(maxlog) && maxlog isa Int
        remaining = get!(logger.message_limits, id, maxlog)
        logger.message_limits[id] = remaining - 1
        remaining > 0 || return nothing
    end

    buf = IOBuffer()
    iob = IOContext(buf, logger.stream)

    level_name = level_to_string(level)
    module_name = something(_module, "nothing")
    file_name   = something(filepath, "nothing")
    line_number = something(line, "nothing")
    msg_timestamp = Dates.format(Dates.now(), "[yyyy/mm/dd HH:MM:SS.sss]")

    crayon = level_to_crayon(level)

    formatted_message = "$(crayon(msg_timestamp)) $(BOLD(crayon(level_name))) $message $(BOLD(crayon("-@->"))) $(UNDERLINE("$file_name:$line_number"))"

    println(iob, formatted_message)
    write(logger.stream, take!(buf))

    return nothing
end

end
