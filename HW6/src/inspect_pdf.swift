// Render all pages with macOS PDFKit and retain extracted text for QA.
import Foundation
import AppKit
import PDFKit

let args = CommandLine.arguments
guard args.count == 3 else { fatalError("Usage: swift inspect_pdf.swift input.pdf output_dir") }
let source = URL(fileURLWithPath: args[1])
let output = URL(fileURLWithPath: args[2])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
guard let document = PDFDocument(url: source), document.pageCount > 0 else {
    fatalError("Cannot open nonempty PDF")
}
var texts: [String] = []
for index in 0..<document.pageCount {
    guard let page = document.page(at: index) else { fatalError("Missing page") }
    let bounds = page.bounds(for: .mediaBox)
    let image = page.thumbnail(of: NSSize(width: bounds.width * 2, height: bounds.height * 2), for: .mediaBox)
    guard let data = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: data),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Cannot render page")
    }
    try png.write(to: output.appendingPathComponent("page-\(index+1).png"))
    texts.append(page.string ?? "")
}
let summary: [String: Any] = ["file": source.lastPathComponent, "page_count": document.pageCount,
                              "page_texts": texts]
try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    .write(to: output.appendingPathComponent("inspection.json"))
print("\(source.lastPathComponent): opened and rendered \(document.pageCount) pages")
