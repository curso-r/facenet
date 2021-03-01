download_url_to_file <- function(url, destfile) {
  p <- utils::download.file(url = url, destfile = destfile)
  invisible(p)
}
