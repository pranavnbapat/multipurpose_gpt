# app/utils/file_utils.py

VIDEO_EXTS  = {"mp4", "avi", "mov", "wmv", "mpeg", "mpg", "mkv", "flv", "webm", "3gp", "mts", "m2ts", "vob", "rmvb"}
TEXT_EXTS   = {"txt", "csv", "tsv", "log", "xml", "json", "yaml", "ini",
               "xls", "xlsx", "ods", "doc", "docx", "odt", "rtf", "ppt", "pptx", "odp",
               "pdf", "epub", "mobi", "azw", "djvu"}
AUDIO_EXTS  = {"mp3", "aac", "wav", "wma", "ogg", "flac", "m4a", "aiff", "opus", "alac", "amr"}
IMAGE_EXTS  = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "heic", "svg", "ico",
               "raw", "nef", "cr2", "psd", "ai", "eps"}
ARCHIVE_EXTS = {"zip", "rar", "7z", "tar", "gz", "bz2", "xz", "zst", "lzma", "iso"}

COMPOUND_ARCHIVE_EXTS = {"tar.gz", "tar.bz2", "tar.xz"}

CATEGORY_BY_EXT = (
    {ext: "video" for ext in VIDEO_EXTS} |
    {ext: "text" for ext in TEXT_EXTS}   |
    {ext: "audio" for ext in AUDIO_EXTS} |
    {ext: "image" for ext in IMAGE_EXTS} |
    {ext: "archive" for ext in ARCHIVE_EXTS}
)

def extract_ext_category(filename: str) -> tuple[str | None, str | None]:
    name = (filename or "").lower()
    for cext in COMPOUND_ARCHIVE_EXTS:
        if name.endswith("." + cext):
            return cext, "archive"

    if "." not in name:
        return None, None

    ext = name.rsplit(".", 1)[-1]
    cat = CATEGORY_BY_EXT.get(ext)
    return ext, cat if cat else (None, None)
