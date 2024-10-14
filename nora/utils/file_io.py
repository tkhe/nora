# Copyright (c) Facebook, Inc. and its affiliates.

import errno
import logging
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import IO
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Union
from urllib import request
from urllib.parse import urlparse

from portalocker import Lock

__all__ = [
    "HTTPURLPathHandler",
    "NativePathHandler",
    "PathHandler",
    "PathManager",
    "download",
    "file_lock",
    "get_cache_dir",
]


def get_cache_dir(cache_dir: str = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $NORA_CACHE, if set
        2) otherwise ~/.torch/nora_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.getenv("NORA_CACHE", "~/.torch/nora_cache"))

    try:
        PathManager.mkdirs(cache_dir)
        assert os.access(cache_dir, os.W_OK)

    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "nora_cache")
        logger = logging.getLogger(__name__)
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        cache_dir = tmp_dir

    return cache_dir


def file_lock(path: str) -> Lock:
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:
        >>> filename = "/path/to/file"
        >>> with file_lock(filename):
        >>>     if not os.path.isfile(filename):
        >>>         do_create_file()
    """
    dirname = os.path.dirname(path)

    try:
        os.makedirs(dirname, exist_ok=True)
    except:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass

    return Lock(path + ".lock", timeout=3600)


def download(url: str, dir: str, *, filename: Optional[str] = None, progress: bool = True) -> str:
    """
    Download a file from a given URL to a directory. If file exists, will not
        overwrite the existing file.

    Args:
        url (str):
        dir (str): the directory to download the file
        filename (str or None): the basename to save the file.
            Will use the name in the URL if not given.
        progress (bool): whether to use tqdm to draw a progress bar.

    Returns:
        str: the path to the downloaded file or the existing one.
    """
    os.makedirs(dir, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]
        # for windows
        if os.name == "nt" and "?" in filename:
            filename = filename[: filename.index("?")]

        assert len(filename), f"Cannot obtain filename from url {url}"

    fpath = os.path.join(dir, filename)
    logger = logging.getLogger(__name__)

    if os.path.isfile(fpath):
        logger.info(f"File {filename} exists! Skipping download.")
        return fpath

    tmp = fpath + ".tmp"  # download to a tmp file first, to be more atomic.
    try:
        logger.info(f"Downloading from {url} ...")
        if progress:
            import tqdm

            def hook(t: tqdm.tqdm) -> Callable[[int, int, Optional[int]], None]:
                last_b: List[int] = [0]

                def inner(b: int, bsize: int, tsize: Optional[int] = None):
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b

                return inner

            with tqdm.tqdm(unit="B", unit_scale=True, miniters=1, desc=filename, leave=True) as t:
                tmp, _ = request.urlretrieve(url, filename=tmp, reporthook=hook(t))

        else:
            tmp, _ = request.urlretrieve(url, filename=tmp)
        statinfo = os.stat(tmp)
        size = statinfo.st_size
        if size == 0:
            raise IOError(f"Downloaded an empty file from {url}!")
        # download to tmp first and move to fpath, to make this function more
        # atomic.
        shutil.move(tmp, fpath)
    except IOError:
        logger.error(f"Failed to download {url}")
        raise
    finally:
        try:
            os.unlink(tmp)
        except IOError:
            pass

    logger.info(f"Successfully downloaded {fpath}. {str(size)} bytes.")
    return fpath


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check: bool = True

    def _check_kwargs(self, kwargs: Dict[str, Any]):
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError(f"Unused arguments: {kwargs}")
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning(f"[PathManager] {k}={v} argument ignored")

    def get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError

    def get_local_path(self, path: str, force: bool = False, **kwargs) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def copy_from_local(self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> Optional[bool]:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def opent(self, path: str, mode: str = "r", buffering: int = 32, **kwargs):
        raise NotImplementedError()

    def open(self, path: str, mode: str = "r", buffering: int = -1, **kwargs) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def copy(self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def mv(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """
        Moves (renames) a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def exists(self, path: str, **kwargs) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def isfile(self, path: str, **kwargs) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def isdir(self, path: str, **kwargs) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def ls(self, path: str, **kwargs) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def mkdirs(self, path: str, **kwargs):
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def rm(self, path: str, **kwargs):
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def symlink(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """
        Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        raise NotImplementedError()

    def set_cwd(self, path: Union[str, None], **kwargs) -> bool:
        """
        Set the current working directory. PathHandler classes prepend the cwd
        to all URI paths that are handled.

        Args:
            path (str) or None: A URI supported by this PathHandler. Must be a valid
                absolute path or None to set the cwd to None.

        Returns:
            bool: true if cwd was set without errors
        """
        raise NotImplementedError()

    def get_path_with_cwd(self, path: str) -> str:
        """
        Default implementation. PathHandler classes that provide a `_set_cwd`
        feature should also override this `_get_path_with_cwd` method.

        Args:
            path (str): A URI supported by this PathHandler.

        Returns:
            path (str): Full path with the cwd attached.
        """
        return path


class NativePathHandler(PathHandler):
    _cwd = None

    def get_local_path(self, path: str, force: bool = False, **kwargs) -> str:
        self._check_kwargs(kwargs)

        return os.fspath(path)

    def copy_from_local(self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> bool:
        self._check_kwargs(kwargs)

        local_path = self.get_path_with_cwd(local_path)
        dst_path = self.get_path_with_cwd(dst_path)

        return self.copy(src_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs)

    def open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: str = None,
        errors: str = None,
        newline: str = None,
        closefd: bool = True,
        opener: Callable = None,
        **kwargs,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a path.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy works as follows:
                    * Binary files are buffered in fixed-size chunks; the size of
                    the buffer is chosen using a heuristic trying to determine the
                    underlying device's “block size” and falling back on
                    io.DEFAULT_BUFFER_SIZE. On many systems, the buffer will
                    typically be 4096 or 8192 bytes long.
            encoding (Optional[str]): the name of the encoding used to decode or
                encode the file. This should only be used in text mode.
            errors (Optional[str]): an optional string that specifies how encoding
                and decoding errors are to be handled. This cannot be used in binary
                mode.
            newline (Optional[str]): controls how universal newlines mode works
                (it only applies to text mode). It can be None, '', '\n', '\r',
                and '\r\n'.
            closefd (bool): If closefd is False and a file descriptor rather than
                a filename was given, the underlying file descriptor will be kept
                open when the file is closed. If a filename is given closefd must
                be True (the default) otherwise an error will be raised.
            opener (Optional[Callable]): A custom opener can be used by passing
                a callable as opener. The underlying file descriptor for the file
                object is then obtained by calling opener with (file, flags).
                opener must return an open file descriptor (passing os.open as opener
                results in functionality similar to passing None).

            See https://docs.python.org/3/library/functions.html#open for details.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)

        return open(
            self.get_path_with_cwd(path),
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    def copy(self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        src_path = self.get_path_with_cwd(src_path)
        dst_path = self.get_path_with_cwd(dst_path)
        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error(f"Destination file {dst_path} already exists.")
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in file copy - {str(e)}")
            return False

    def mv(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """
        Moves (renames) a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        src_path = self.get_path_with_cwd(src_path)
        dst_path = self.get_path_with_cwd(dst_path)
        if os.path.exists(dst_path):
            logger = logging.getLogger(__name__)
            logger.error(f"Destination file {dst_path} already exists.")
            return False

        try:
            shutil.move(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in move operation - {str(e)}")
            return False

    def symlink(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """
        Creates a symlink to the src_path at the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        src_path = self.get_path_with_cwd(src_path)
        dst_path = self.get_path_with_cwd(dst_path)

        logger = logging.getLogger(__name__)
        if not os.path.exists(src_path):
            logger.error(f"Source path {src_path} does not exist")
            return False

        if os.path.exists(dst_path):
            logger.error(f"Destination path {dst_path} already exists.")
            return False

        try:
            os.symlink(src_path, dst_path)
            return True
        except Exception as e:
            logger.error(f"Error in symlink - {str(e)}")
            return False

    def exists(self, path: str, **kwargs) -> bool:
        self._check_kwargs(kwargs)

        return os.path.exists(self.get_path_with_cwd(path))

    def isfile(self, path: str, **kwargs) -> bool:
        self._check_kwargs(kwargs)

        return os.path.isfile(self.get_path_with_cwd(path))

    def isdir(self, path: str, **kwargs) -> bool:
        self._check_kwargs(kwargs)

        return os.path.isdir(self.get_path_with_cwd(path))

    def ls(self, path: str, **kwargs) -> List[str]:
        self._check_kwargs(kwargs)

        return os.listdir(self.get_path_with_cwd(path))

    def mkdirs(self, path: str, **kwargs):
        self._check_kwargs(kwargs)

        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def rm(self, path: str, **kwargs):
        self._check_kwargs(kwargs)

        os.remove(path)

    def set_cwd(self, path: Union[str, None], **kwargs) -> bool:
        self._check_kwargs(kwargs)

        # Remove cwd path if None
        if path is None:
            self._cwd = None
            return True

        # Make sure path is a valid Unix path
        if not os.path.exists(path):
            raise ValueError(f"{path} is not a valid Unix path")

        # Make sure path is an absolute path
        if not os.path.isabs(path):
            raise ValueError(f"{path} is not an absolute path")

        self._cwd = path
        return True

    def get_path_with_cwd(self, path: str) -> str:
        if not path:
            return path

        return os.path.normpath(path if not self._cwd else os.path.join(self._cwd, path))


class HTTPURLPathHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    MAX_FILENAME_LEN = 250

    def __init__(self):
        self.cache_map: Dict[str, str] = {}

    def get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def get_local_path(
        self,
        path: str,
        force: bool = False,
        cache_dir: str = None,
        **kwargs,
    ) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)

        if (
            force
            or path not in self.cache_map
            or not os.path.exists(self.cache_map[path])
        ):
            logger = logging.getLogger(__name__)

            parsed_url = urlparse(path)
            dirname = os.path.join(get_cache_dir(cache_dir), os.path.dirname(parsed_url.path.lstrip("/")))
            filename = path.split("/")[-1]

            if parsed_url.query:
                filename = filename.split("?").pop(0)

            if len(filename) > self.MAX_FILENAME_LEN:
                filename = filename[:100] + "_" + uuid.uuid4().hex

            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info(f"Downloading {path} ...")
                    cached = download(path, dirname, filename=filename)
            logger.info(f"URL {path} cached in {cached}")
            self.cache_map[path] = cached
        return self.cache_map[path]

    def open(self, path: str, mode: str = "r", buffering: int = -1, **kwargs) -> Union[IO[str], IO[bytes]]:
        """
        Open a remote HTTP path. The resource is first downloaded and cached
        locally.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)

        assert mode in ("r", "rb"), f"{self.__class__.__name__} does not support open with {mode} mode"
        assert buffering == -1, f"{self.__class__.__name__} does not support the `buffering` argument"

        local_path = self.get_local_path(path, force=False)
        return open(local_path, mode)


class PathManagerBase:
    """
    A class for users to open generic paths or translate generic paths to file names.

    path_manager.method(path) will do the following:
    1. Find a handler by checking the prefixes in `self._path_handlers`.
    2. Call handler.method(path) on the handler that's found
    """

    def __init__(self):
        self._path_handlers: MutableMapping[str, PathHandler] = OrderedDict()
        """
        Dict from path prefix to handler.
        """

        self._native_path_handler: PathHandler = NativePathHandler()
        """
        A NativePathHandler that works on posix paths. This is used as the fallback.
        """

        self._cwd: Optional[str] = None
        """
        Keeps track of the single cwd (if set).
        NOTE: Only one PathHandler can have a cwd set at a time.
        """

    def get_path_handler(self, path: str) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str): URI path to resource

        Returns:
            handler (PathHandler)
        """
        path = os.fspath(path)
        for p, handler in self._path_handlers.items():
            if path.startswith(p):
                return handler
        return self._native_path_handler

    def opent(self, path: str, mode: str = "r", buffering: int = 32, **kwargs):
        """
        Open a tabular data source. Only reading is supported.
        The opent() returns a Python iterable collection object, compared to bytes/text data with open()

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'
            buffering (int): number of rows fetched and cached

        Returns:
            A TabularIO context manager object
        """
        return self.get_path_handler(path).opent(path, mode, buffering, **kwargs)

    def open(self, path: str, mode: str = "r", buffering: int = -1, **kwargs) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        return self.get_path_handler(path).open(path, mode, buffering=buffering, **kwargs)

    def copy(self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        handler = self.get_path_handler(src_path)

        if handler != self.get_path_handler(dst_path):
            return self.copy_across_handlers(src_path, dst_path, overwrite, **kwargs)

        return handler.copy(src_path, dst_path, overwrite, **kwargs)

    def mv(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """
        Moves (renames) a source path supported by NativePathHandler to
        a destination path.

        Args:
            src_path (str): A URI supported by NativePathHandler
            dst_path (str): A URI supported by NativePathHandler

        Returns:
            status (bool): True on success
        Exception:
            Asserts if both the src and dest paths are not supported by
            NativePathHandler.
        """
        handler = self.get_path_handler(src_path)

        assert handler == self.get_path_handler(dst_path), "Src and dest paths must be supported by the same path handler."

        return handler.mv(src_path, dst_path, **kwargs)

    def get_local_path(self, path: str, force: bool = False, **kwargs) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        path = os.fspath(path)
        handler = self.get_path_handler(path)

        try:
            return handler.get_local_path(path, force=force,  **kwargs)
        except TypeError:
            return handler.get_local_path(path, **kwargs)

    def copy_from_local(self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> bool:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        assert os.path.exists(local_path), f"local_path = {local_path}"

        return self.get_path_handler(dst_path).copy_from_local(local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs)

    def exists(self, path: str, **kwargs) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return self.get_path_handler(path).exists(path, **kwargs)

    def isfile(self, path: str, **kwargs) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return self.get_path_handler(path).isfile(path, **kwargs)

    def isdir(self, path: str, **kwargs) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return self.get_path_handler(path).isdir(path, **kwargs)

    def ls(self, path: str, **kwargs) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return self.get_path_handler(path).ls(path, **kwargs)

    def mkdirs(self, path: str, **kwargs):
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return self.get_path_handler(path).mkdirs(path, **kwargs)

    def rm(self, path: str, **kwargs):
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return self.get_path_handler(path).rm(path, **kwargs)

    def symlink(self, src_path: str, dst_path: str, **kwargs) -> bool:
        """Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        # Copying across handlers is not supported.
        handler = self.get_path_handler(src_path)
        assert handler == self.get_path_handler(dst_path)

        return handler.symlink(src_path, dst_path, **kwargs)

    def set_cwd(self, path: Union[str, None], **kwargs) -> bool:
        """
        Set the current working directory. PathHandler classes prepend the cwd
        to all URI paths that are handled.

        Args:
            path (str) or None: A URI supported by this PathHandler. Must be a valid
                absolute Unix path or None to set the cwd to None.

        Returns:
            bool: true if cwd was set without errors
        """
        if path is None and self._cwd is None:
            return True

        if self.get_path_handler(path or self._cwd).set_cwd(path, **kwargs):
            self._cwd = path
            return True
        else:
            return False

    def register_handler(self, handler: PathHandler, allow_override: bool = True):
        """
        Register a path handler associated with `handler.get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
            allow_override (bool): allow overriding existing handler for prefix
        """
        assert isinstance(handler, PathHandler), handler

        # Allow override of `NativePathHandler` which is automatically
        # instantiated by `PathManager`.
        if isinstance(handler, NativePathHandler):
            if allow_override:
                self._native_path_handler = handler
            else:
                raise ValueError("`NativePathHandler` is registered by default. Use the `allow_override=True` kwarg to override it.")
            return

        for prefix in handler.get_supported_prefixes():
            if prefix not in self._path_handlers:
                self._path_handlers[prefix] = handler
                continue

            old_handler_type = type(self._path_handlers[prefix])
            if allow_override:
                self._path_handlers[prefix] = handler
            else:
                raise KeyError(f"[PathManager] Prefix '{prefix}' already registered by {old_handler_type}!")

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        self._path_handlers = OrderedDict(sorted(self._path_handlers.items(), key=lambda t: t[0], reverse=True))

    def set_strict_kwargs_checking(self, enable: bool):
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.

        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.

        Args:
            enable (bool)
        """
        self._native_path_handler._strict_kwargs_check = enable
        for handler in self._path_handlers.values():
            handler._strict_kwargs_check = enable

    def copy_across_handlers(self, src_path: str, dst_path: str, overwrite: bool, **kwargs: Any) -> bool:
        src_handler = self.get_path_handler(src_path)
        dst_handler = self.get_path_handler(dst_path)

        assert src_handler.get_local_path is not None
        assert dst_handler.copy_from_local is not None

        local_file = src_handler.get_local_path(src_path, **kwargs)
        return dst_handler.copy_from_local(local_file, dst_path, overwrite=overwrite, **kwargs)


PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLPathHandler())
