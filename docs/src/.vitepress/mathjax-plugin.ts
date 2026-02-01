// adapter from https://github.com/orgs/vuepress-theme-hope/discussions/5178#discussioncomment-15642629
// mathjax-plugin.ts
// @ts-ignore
import MathJax from '@mathjax/src'
import type { Plugin as VitePlugin } from 'vite'
import type MarkdownIt from 'markdown-it'
import { tex as mdTex } from '@mdit/plugin-tex'

const mathjaxStyleModuleID = 'virtual:mathjax-styles.css'

interface MathJaxOptions {
  font?: string
}

async function initializeMathJax(options: MathJaxOptions = {}) {
  const font = options.font || 'mathjax-newcm'

  const config: any = {
    loader: {
      load: [
        'input/tex',
        'output/svg',
        '[tex]/boldsymbol',
        '[tex]/braket',
        '[tex]/mathtools',
      ],
    },
    tex: {
      tags: 'ams',
      packages: {
        '[+]': ['boldsymbol', 'braket', 'mathtools'],
      },
    },
    output: {
      font,
      displayOverflow: 'linebreak',
      mtextInheritFont: true,
    },
    svg: {
      fontCache: 'none', // critical: avoids async font loading
    },
  }

  await MathJax.init(config)

  const fontData = MathJax.config.svg?.fontData

  if (fontData?.dynamicFiles) {
    const dynamicFiles = fontData.dynamicFiles
    const dynamicPrefix: string =
      fontData.OPTIONS?.dynamicPrefix || fontData.options?.dynamicPrefix

    if (dynamicPrefix) {
      await Promise.all(
        Object.keys(dynamicFiles).map(async (name) => {
          try {
            await import(/* @vite-ignore */ `${dynamicPrefix}/${name}.js`)
            dynamicFiles[name]?.setup?.(MathJax.startup.output.font)
          } catch {
            // Silently ignore missing dynamic files
          }
        }),
      )
    }
  }
}

export function mathjaxPlugin(options: MathJaxOptions = {}) {
  let adaptor: any
  let initialized = false

  async function ensureInitialized() {
    if (!initialized) {
      await initializeMathJax(options)
      adaptor = MathJax.startup.adaptor
      initialized = true
    }
  }

  function renderMath(content: string, displayMode: boolean): string {
    if (!initialized) {
      throw new Error('MathJax not initialized')
    }

    const node = MathJax.tex2svg(content, { display: displayMode })

    // Prevent Vue from touching MathJax output
    adaptor.setAttribute(node, 'v-pre', '')

    let html = adaptor.outerHTML(node)

    // Preserve spaces inside mjx-break (SVG only)
    html = html.replace(
      /<mjx-break(.*?)>(.*?)<\/mjx-break>/g,
      (_: string, attr: string, inner: string) =>
        `<mjx-break${attr}>${inner.replace(/ /g, '&nbsp;')}</mjx-break>`,
    )
    
    // Wrap only display equations (not inline math)
    html = html.replace(
      /(<mjx-container[^>]*display="true"[^>]*>)([\s\S]*?)(<\/mjx-container>)/,
      '<div class="mjx-scroll-wrapper">$1$2$3</div>'
    )

    return html
  }

  function getMathJaxStyles(): string {
    return initialized
      ? adaptor.textContent(MathJax.svgStylesheet()) || ''
      : ''
  }

  function resetMathJax(): void {
    if (!initialized) return
    MathJax.texReset()
    MathJax.typesetClear()
  }

  function viteMathJax(): VitePlugin {
    const virtualModuleID = '\0' + mathjaxStyleModuleID

    return {
      name: 'mathjax-styles',

      resolveId(id) {
        if (id === mathjaxStyleModuleID) {
          return virtualModuleID
        }
      },

      async load(id) {
        if (id === virtualModuleID) {
          await ensureInitialized()
          return getMathJaxStyles()
        }
      },
    }
  }

  function mdMathJax(md: MarkdownIt): void {
    mdTex(md, {
      render: renderMath,
    })

    const orig = md.render
    md.render = function (...args) {
      resetMathJax()
      return orig.apply(this, args)
    }
  }

  const init = ensureInitialized()

  return {
    vitePlugin: viteMathJax(),
    markdownConfig: mdMathJax,
    styleModuleID: mathjaxStyleModuleID,
    init,
  }
}