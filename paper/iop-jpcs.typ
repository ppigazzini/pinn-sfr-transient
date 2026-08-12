// IOP Journal of Physics: Conference Series layout, from __DEV/paper_template.docx.
//
// The template is a Word layout guide; this reproduces its *specification* rather
// than its file format. Every constant below is taken from its Table 1 (margins)
// and Table 2 (section styles) so the mapping is checkable rather than eyeballed.

#let jpcs(
  title: none,
  authors: (),
  affiliations: (),
  email: none,
  abstract: none,
  body,
) = {
  set document(title: title, author: authors.map(a => a.name))
  // Table 1 of the guide: A4 only, top 4.0, bottom 2.7, left 2.5, right 2.5 cm,
  // and no headers, footers or page numbers -- production adds those.
  set page(paper: "a4", margin: (top: 4.0cm, bottom: 2.7cm, left: 2.5cm, right: 2.5cm))
  set text(font: ("Times New Roman", "Nimbus Roman", "TeX Gyre Termes"), size: 11pt)
  set par(justify: true, leading: 0.55em, first-line-indent: 5mm)
  set heading(numbering: "1.1.1.")

  // Table 2: sections 11pt bold, subsections 11pt italic, one line space before,
  // none after. Typst's `show` rules express that directly.
  show heading.where(level: 1): it => {
    v(1em, weak: true)
    text(size: 11pt, weight: "bold")[#counter(heading).display(). #it.body]
    v(0em, weak: true)
  }
  show heading.where(level: 2): it => {
    v(1em, weak: true)
    text(size: 11pt, style: "italic")[#counter(heading).display() #it.body]
    v(0em, weak: true)
  }
  // Subsubsections end with a full stop and run into the paragraph.
  show heading.where(level: 3): it => {
    text(size: 11pt, style: "italic")[#counter(heading).display() #it.body.]
    h(0.4em)
  }

  // Title: 17pt Times bold, flush left, unjustified.
  block(width: 100%)[
    #set par(justify: false, first-line-indent: 0mm)
    #text(size: 17pt, weight: "bold")[#title]
  ]
  v(1.2em)

  // Authors indented 25 mm to match the abstract.
  block(width: 100%, inset: (left: 25mm))[
    #set par(justify: false, first-line-indent: 0mm)
    #authors.enumerate().map(((i, a)) => [#a.name#super[#a.aff]]).join(", ")
  ]
  v(0.6em)
  block(width: 100%, inset: (left: 25mm))[
    #set par(justify: false, first-line-indent: 0mm)
    #set text(size: 10pt)
    #affiliations.enumerate().map(((i, s)) => [#super[#(i + 1)] #s]).join(linebreak())
    #if email != none [#linebreak() Corresponding author e-mail: #email]
  ]
  v(1.2em)

  // Abstract: 10pt, indented 25 mm from the LEFT margin only. The guide says left; an
  // earlier revision indented both sides, narrowing the measure by a further 25 mm.
  block(width: 100%, inset: (left: 25mm))[
    #set text(size: 10pt)
    #set par(justify: true, first-line-indent: 0mm)
    *Abstract.* #abstract
  ]
  v(1.5em)

  set figure(numbering: "1")
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: image): set figure.caption(position: bottom)
  show figure.caption: set text(size: 10pt)

  body
}

// IOP numeric reference list: [n] Authors Year Title Journal Volume Page.
#let references(items) = {
  heading(numbering: none)[References]
  // Guide: 5 mm gap after the number, 5 mm indent on continuation lines. Was 8 mm.
  set par(first-line-indent: 0mm, hanging-indent: 5mm)
  set text(size: 10pt)
  for (i, r) in items.enumerate() {
    block(spacing: 0.5em)[#box(width: 5mm)[\[#(i + 1)\]] #r]
  }
}
