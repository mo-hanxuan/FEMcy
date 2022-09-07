"""
use a branch thread to show the progress bar, 
in order to indicate that the main thread is computing,
function stop_thread(threadID) can stop the thread
API is:
    from progressBar import *
"""

import threading
import time, sys
import inspect
import ctypes


def progressBar_percentage(percentage=0):
    halfPercent = int(percentage / 2)
    sys.stdout.write("||{}{}|| -> {:5.3f} % \r".format(
        '#' * halfPercent, 
        '-' * (50 - halfPercent), 
        percentage
    ))
    sys.stdout.flush()

 
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
 
 
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
 
 
def progressBar():
    tBeg = time.time()
    j = '#'
    while True:
        j += '#'
        sys.stdout.write(' ||' + j + ' -> time = ' + str(int(time.time() - tBeg)) + 's' + "\r")
        sys.stdout.flush()
        time.sleep(1.)
 
 
if __name__ == "__main__":
    t = threading.Thread(target=progressBar)
    t.start()
    time.sleep(20.)
    print("main thread sleep finish")
    stop_thread(t)
    t.join()
    print('branch thread stop !!!!')