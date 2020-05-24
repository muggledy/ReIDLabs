# coding:utf-8

"""Python Terminal's colorful output"""

import ctypes, sys
from functools import partial

__all__ = ['cprint', 'cprint_out', 'cprint_err', 'cprint_in', '__STDOUT__', '__STDERR__', '__STDIN__','fcolors','bcolors']

# This code modified from: https://blog.csdn.net/wy_97/article/details/79663014

STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12

# foreground colors
FOREGROUND_BLACK = 0x00
FOREGROUND_DARKBLUE = 0x01
FOREGROUND_DARKGREEN = 0x02
FOREGROUND_DARKSKYBLUE = 0x03
FOREGROUND_DARKRED = 0x04
FOREGROUND_DARKPINK = 0x05
FOREGROUND_DARKYELLOW = 0x06
FOREGROUND_DARKWHITE = 0x07
FOREGROUND_DARKGRAY = 0x08
FOREGROUND_BLUE = 0x09
FOREGROUND_GREEN = 0x0a
FOREGROUND_SKYBLUE = 0x0b
FOREGROUND_RED = 0x0c
FOREGROUND_PINK = 0x0d
FOREGROUND_YELLOW = 0x0e
FOREGROUND_WHITE = 0x0f

# background colors
BACKGROUND_BLUE = 0x10
BACKGROUND_GREEN = 0x20
BACKGROUND_DARKSKYBLUE = 0x30
BACKGROUND_DARKRED = 0x40
BACKGROUND_DARKPINK = 0x50
BACKGROUND_DARKYELLOW = 0x60
BACKGROUND_DARKWHITE = 0x70
BACKGROUND_DARKGRAY = 0x80
BACKGROUND_BLUE = 0x90
BACKGROUND_GREEN = 0xa0
BACKGROUND_SKYBLUE = 0xb0
BACKGROUND_RED = 0xc0
BACKGROUND_PINK = 0xd0
BACKGROUND_YELLOW = 0xe0
BACKGROUND_WHITE = 0xf0

# backup of standard stream
__STDOUT__ = sys.stdout
__STDERR__ = sys.stderr
__STDIN__ = sys.stdin

# get handle
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
std_err_handle = ctypes.windll.kernel32.GetStdHandle(STD_ERROR_HANDLE)
std_in_handle = ctypes.windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
HANDLES = [std_out_handle, std_err_handle, std_in_handle]

# set color
def set_cmd_text_color(color, handle = std_out_handle):
    return ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)

# reset white
resetColor = partial(set_cmd_text_color, color = FOREGROUND_WHITE)

def cprint(*texts, fcolor = None, bcolor = None, **kwargs):
    '''
       Print Texts Colorfully
       
       Supported foreground color(arg fcolor): 
          black, darkblue, darkgreen, darkskyblue, darkred, darkpink, darkyellow, darkwhite, 
          darkgray, blue, green, skyblue, red, pink, yellow, white
       Supported background color(arg bcolor):
          darkblue, darkgreen, darkskyblue, darkreed, darkpink, darkyellow, darkwhite, 
          darkgray, blue, green, skyblue, red, pink, yellow, white
       
       Note that texts and kwargs will be wholely passed to func print, herein i won't explain 
       which parameters are available about kwargs, you can see help doc of func print. You 
       have to set the parameters repeatedly every time when you cprint, because the color setting 
       is always reseted at the end, for this reason, we provide several partial functions that 
       fixed color settings:
          >>> cprint_out=partial(cprint,fcolor='green')
          >>> cprint_err=partial(cprint,fcolor='white',bcolor='red',file=sys.stderr)
          >>> cprint_in=partial(cprint,fcolor='blue',file=sys.stdin,end=' ')
       
       If you don't set fcolor and bolcor, cprint is equal to print in this case
       
       This function is only suitable for Windows CMD
    '''
    fcolor = None if fcolor == None else globals().get('FOREGROUND_' + fcolor.upper())
    bcolor = None if bcolor == None else globals().get('BACKGROUND_' + bcolor.upper())
    c = (fcolor | bcolor) if (fcolor and bcolor) else (fcolor or bcolor or FOREGROUND_WHITE)
    file = kwargs.get('file')
    index = 0
    if file == sys.stderr:
        index = 1
    elif file == sys.stdin:
        index = 2
        del kwargs['file']
    kwargs['flush'] = True
    set_cmd_text_color(c, HANDLES[0] if index == 2 else HANDLES[index])
    print(*texts, **kwargs)
    ret = None
    if index == 2:
        ret = file.readline()
    resetColor(handle = HANDLES[0] if index == 2 else HANDLES[index])
    if ret != None:
        return ret.replace('\n','').replace('\r','')

cprint_out = partial(cprint, fcolor = 'green')
cprint_err = partial(cprint, fcolor = 'white', bcolor = 'red', file = sys.stderr)
cprint_in = partial(cprint, fcolor = 'blue', file = sys.stdin, end = ' ')

fcolors=['red', 'blue', 'green', 'pink', 'yellow', 'darkred', 'darkblue', 'darkgreen', 'darkpink', 
         'darkyellow', 'darkskyblue', 'darkwhite', 'skyblue', 'white', 'darkgray', 'black']
bcolors=['darkblue', 'darkgreen', 'darkskyblue', 'darkreed', 'darkpink', 'darkyellow', 'darkwhite', 
         'darkgray', 'blue', 'green', 'skyblue', 'red', 'pink', 'yellow', 'white']

if __name__ == '__main__':
    cprint_out('Hi,', 'Muggledy!')
    cprint('Hi,', 'Muggledy!', fcolor = 'darkblue')
    cprint('Hi,', 'Muggledy!', fcolor = 'black', bcolor = 'white')
    cprint('Hi,', 'Muggledy!', fcolor = 'red', bcolor = 'yellow')
    cprint('Hi,', 'Muggledy!')
    cprint_err('Hi,', 'Muggledy!')
    cprint('Hi,', 'Muggledy!', fcolor = 'pink', file = sys.stderr)
    #s = cprint_in('输入：')
    #cprint_out(s)
