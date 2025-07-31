import os
import time
import argparse
from typing import Callable, Never, Optional, Generator
from prometheus_client import start_http_server
from . import metrics
from dataclasses import dataclass

type Timestamp = float
type Seconds = float

@dataclass
class DirInfo:
    path: str
    size: int
    latest_mtime: Timestamp
    oldest_mtime: Timestamp
    entries_count: int
    processing_time: Seconds

ONE_S_IN_NS = 1_000_000_000


class BudgetedDirInfoWalker:
    def __init__(self, iops_budget: int=100):
        """
        iops_budget is number of io operations allowed every second.
        """
        self.iops_budget = iops_budget
        # Use _ns to avoid subtractive messiness possible when using floats
        self._last_iops_reset_time = time.monotonic_ns()
        self._io_calls_since_last_reset = 0

    def do_iops_action[R, **P](self, func: Callable[P, R], *args, **kwargs) -> R:
        """
        Perform an action that does IO, waiting if necessary so it is within budget.

        All IO performed should be wrapped with this function, so we do not exceed
        our budget. Each call to this function is treated as one IO.
        """
        if time.monotonic_ns() - self._last_iops_reset_time > ONE_S_IN_NS:
            # One second has passed since last time we reset the budget clock
            # So we reset it again now, regardless of how many iops have happened
            self._io_calls_since_last_reset = 0
            self._last_iops_reset_time = time.monotonic_ns()

        if self._io_calls_since_last_reset > self.iops_budget:
            # We are over budget, so we wait for 1s + 1ns since last reset
            # IO can be performed once this is wait is done. We reset the budget clock
            # after our wait.
            wait_period_in_s = (
                ONE_S_IN_NS - (time.monotonic_ns() - self._last_iops_reset_time) + 1
            ) / ONE_S_IN_NS
            time.sleep(wait_period_in_s)
            self._io_calls_since_last_reset = 0
            self._last_iops_reset_time = time.monotonic_ns()

        return_value = func(*args, **kwargs)
        self._io_calls_since_last_reset += 1
        return return_value

    def find_urb_directory(self, base_path: str) -> Optional[str]:
        """
        Find the first subdirectory that contains a .urb subdirectory.
        Only used for tlon_ volumes.
        """
        try:
            children = [
                os.path.join(base_path, c)
                for c in self.do_iops_action(os.listdir, base_path)
            ]
            
            dirs = [c for c in children if self.do_iops_action(os.path.isdir, c)]
            
            for d in dirs:
                urb_path = os.path.join(d, '.urb')
                if self.do_iops_action(os.path.isdir, urb_path):
                    return d
            
            return None
        except (OSError, PermissionError):
            return None

    def get_dir_info(self, path: str) -> Optional[DirInfo]:
        start_time = time.monotonic()
        try:
            self_statinfo = self.do_iops_action(os.stat, path)
        except FileNotFoundError:
            # Directory was deleted from the time it was listed and now
            return None

        # Get absolute path of all children of directory
        children = [
            os.path.abspath(os.path.join(path, c))
            for c in self.do_iops_action(os.listdir, path)
        ]
        # Split into files and directories for different kinds of traversal.
        # We count symlinks as files, but do not resolve them when checking size -
        # but do include them in the mtime calculation.
        files = [
            c
            for c in children
            if self.do_iops_action(os.path.isfile, c)
            or self.do_iops_action(os.path.islink, c)
        ]
        dirs = [c for c in children if self.do_iops_action(os.path.isdir, c)]

        total_size = self_statinfo.st_size
        latest_mtime = self_statinfo.st_mtime
        oldest_mtime = self_statinfo.st_mtime
        entries_count = len(files) + 1  # Include this directory as an entry

        for f in files:
            # Do not follow symlinks, as that may lead to double counting a symlinked
            # file's size.
            try:
                stat_info = self.do_iops_action(os.stat, f, follow_symlinks=False)
            except FileNotFoundError:
                # File might have been deleted from the time we listed it and now
                continue
            total_size += stat_info.st_size
            if latest_mtime < stat_info.st_mtime:
                latest_mtime = stat_info.st_mtime
            if oldest_mtime > stat_info.st_mtime:
                oldest_mtime = stat_info.st_mtime

        for d in dirs:
            dirinfo = self.get_dir_info(d)
            if dirinfo is None:
                # The directory was deleted between the time the listing
                # was done and now.
                continue
            total_size += dirinfo.size
            entries_count += dirinfo.entries_count
            if latest_mtime < dirinfo.latest_mtime:
                latest_mtime = dirinfo.latest_mtime
            if oldest_mtime > dirinfo.oldest_mtime:
                oldest_mtime = dirinfo.oldest_mtime

        return DirInfo(
            path=os.path.basename(path),
            size=total_size,
            latest_mtime=latest_mtime,
            oldest_mtime=oldest_mtime,
            entries_count=entries_count,
            processing_time=time.monotonic() - start_time,
        )

    def get_subdirs_info(self, dir_path: str) -> Generator[tuple[str, DirInfo] | None, None, None]:
        try:
            children = [
                os.path.abspath(os.path.join(dir_path, c))
                for c in self.do_iops_action(os.listdir, dir_path)
            ]

            dirs = [c for c in children if self.do_iops_action(os.path.isdir, c)]

            for c in dirs:
                dir_name = os.path.basename(c)
                
                # For tlon_ volumes, look for .urb subdirectory
                if "tlon_" in dir_name:
                    urb_dir = self.find_urb_directory(c)
                    if urb_dir:
                        dirinfo = self.get_dir_info(urb_dir)
                        if dirinfo:
                            # Use the original directory name but measure the urb subdirectory
                            yield (dir_name, dirinfo)
                    else:
                        # No .urb found, measure the whole directory
                        dirinfo = self.get_dir_info(c)
                        if dirinfo:
                            yield (dir_name, dirinfo)
                else:
                    # Non-tlon volumes, measure normally
                    dirinfo = self.get_dir_info(c)
                    if dirinfo:
                        yield (dir_name, dirinfo)
        except OSError as e:
            if e.errno == 116:
                # See https://github.com/yuvipanda/prometheus-dirsize-exporter/issues/6
                # Stale file handle, often because the file we were looking at
                # changed in the NFS server via another client in such a way that
                # a new inode was created. This is a race, so let's just ignore and
                # not report any data for this file. If this file was recreated,
                # our next run should catch it
                return None
            # Any other errors should just be propagated
            raise
        except PermissionError as e:
            if e.errno == 13:
                # See https://github.com/yuvipanda/prometheus-dirsize-exporter/issues/5
                # A file we are trying to open is owned in such a way that we don't have
                # access to it. Ideally this should not really happen, but when it does,
                # we just ignore it and continue.
                return None
            # Any other permission error should be propagated
            raise

def main() -> Never:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "parent_dir",
        help="The directory to whose subdirectories will have their information exported",
    )
    argparser.add_argument(
        "iops_budget", help="Number of IO operations allowed per second", type=int
    )
    argparser.add_argument(
        "wait_time_minutes",
        help="Number of minutes to wait before data collection runs",
        type=int,
    )
    # Don't report amount of time it took to process each directory by
    # default. This is highly variable, and probably causes prometheus to
    # not compress metrics very well. Not particularly useful outside of
    # debugging the exporter itself.
    argparser.add_argument(
        "--enable-detailed-processing-time-metric",
        help="Report amount of time it took to process each directory",
        action="store_true"
    )
    argparser.add_argument(
        "--port", help="Port for the server to listen on", type=int, default=8000
    )

    args = argparser.parse_args()

    start_http_server(args.port)
    
    # track currently active directories to clean up stale metrics
    active_directories = set()
    
    while True:
        walker = BudgetedDirInfoWalker(args.iops_budget)
        current_directories = set()
        
        for result in walker.get_subdirs_info(args.parent_dir):
            if result is None:
                continue
                
            dir_name, subdir_info = result
            current_directories.add(dir_name)
            
            metrics.TOTAL_SIZE.labels(dir_name).set(subdir_info.size)
            metrics.LATEST_MTIME.labels(dir_name).set(subdir_info.latest_mtime)
            metrics.OLDEST_MTIME.labels(dir_name).set(subdir_info.oldest_mtime)
            metrics.ENTRIES_COUNT.labels(dir_name).set(subdir_info.entries_count)
            if args.enable_detailed_processing_time_metric:
                metrics.PROCESSING_TIME.labels(dir_name).set(subdir_info.processing_time)
            metrics.LAST_UPDATED.labels(dir_name).set(time.time())
            print(f"Updated values for {dir_name}")
        
        # clean up metrics for directories that no longer exist
        stale_directories = active_directories - current_directories
        for stale_dir in stale_directories:
            try:
                metrics.TOTAL_SIZE.remove(stale_dir)
                metrics.LATEST_MTIME.remove(stale_dir)
                metrics.OLDEST_MTIME.remove(stale_dir)
                metrics.ENTRIES_COUNT.remove(stale_dir)
                metrics.LAST_UPDATED.remove(stale_dir)
                if args.enable_detailed_processing_time_metric:
                    metrics.PROCESSING_TIME.remove(stale_dir)
                print(f"Cleaned up stale metrics for {stale_dir}")
            except KeyError:
                pass
        
        active_directories = current_directories
        time.sleep(args.wait_time_minutes * 60)


if __name__ == "__main__":
    main()