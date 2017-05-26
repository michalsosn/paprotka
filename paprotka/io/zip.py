import zipfile as zf


def walk_zip(path, key=lambda info: info.filename, *args, **kwargs):
    with zf.ZipFile(path) as zipfile:
        for info in sorted(zipfile.infolist(), key=key):
            name = info.filename
            if not name.endswith('/'):
                yield zipfile.open(name, *args, **kwargs)


def walk_zip_names(path, filter=lambda name: True, key=lambda info: info.filename):
    with zf.ZipFile(path) as zipfile:
        for info in sorted(zipfile.infolist(), key=key):
            name = info.filename
            if not name.endswith('/') and filter(name):
                yield name
