using Documenter
using Documenter.Anchors
using Documenter.Documents
using Documenter.Expanders
using Documenter.Selectors

using Bibliography: xnames, xyear, xlink, xtitle, xin

abstract type BibliographyBlock <: Expanders.ExpanderPipeline end

Selectors.order(::Type{BibliographyBlock}) = 12.0  # Expand bibliography last
Selectors.matcher(::Type{BibliographyBlock}, node, page, doc) = Expanders.iscode(node, r"^@bibliography")

function Selectors.runner(::Type{BibliographyBlock}, x, page, doc)
    @info "Expanding bibliography."
    raw_bib = "<dl>"
    for (id, entry) in BIBLIOGRAPHY
        @info "Expanding bibliography entry: $id."

        # Add anchor that citations can link to from anywhere in the docs.
        Anchors.add!(doc.internal.headers, entry, entry.id, page.build)

        entry_text = """<dt>$id</dt>
        <dd>
          <div id="$id">$(xnames(entry)) ($(xyear(entry))), <a href="$(xlink(entry))">$(xtitle(entry))</a>, $(xin(entry))</a>
        </dd>"""
        raw_bib *= entry_text
    end
    raw_bib *= "\n</dl>"

    page.mapping[x] = Documents.RawNode(:html, raw_bib)
end
