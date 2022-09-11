# -*- coding: utf-8 -*-

# -------------------------------------------
# Module
# get the color by a specific mod
# -------------------------------------------

import warnings


def getColor(x, mod=4):
    delta = 1.e-3
    if not (0. - delta <= x <= 1. + delta):
        red, green, blue = 0.2, 0.2, 0.2
        if x > 1. + delta:
            red, green, blue = 0.5, 0.5, 0.5
            warnings.warn('colorBar x > 1.')
        elif x < 0. - delta:
            red, green, blue = 0.2, 0.2, 0.2
            warnings.warn('colorBar x < 0.')
        return red, green, blue
    def case1():
        # (1.0 red -> 0.5 green -> 0.0 blue)
        if x >= 0.5:
            red = (x - 0.5) / 0.5
            green = (1. - x) / 0.5
            blue = 0.
        else:
            red = 0.
            green = x / 0.5
            blue = (0.5 - x) / 0.5
        return red, green, blue
    def case2():
        # (1, 0, 0) -> (0.5, 1, 0.5) -> (0, 0, 1)
        #   red     ->  bright green -> blue (more smooth)
        red = x
        blue = 1. - x
        green = (1. - x) / 0.5 if x >= 0.5 else x / 0.5
        return red, green, blue
    def case3():
        # (1.0 red -> 0.5 white -> 0.0 blue)
        if x >= 0.5:
            red = 1.
            green = (1. - x) / 0.5
            blue = (1. - x) / 0.5
        else:
            red = x / 0.5
            green = x / 0.5
            blue = 1.
        return red, green, blue
    def case4():
        # rainbow colorBar, 4 intervals,
        #      (1 ~  0.75  ~ 0.5   ~ 0.25 ~ 0)
        # -> (red ~ yellow ~ green ~ cyan ~ blue)
        if x >= 0.75:
            red = 1.
            green = (1. - x) / 0.25
            blue = 0.
        elif 0.5 <= x < 0.75:
            red = (x - 0.5) / 0.25
            green = 1.
            blue = 0.
        elif 0.25 <= x < 0.5:
            red = 0.
            green = 1.
            blue = (0.5 - x) / 0.25
        else:
            red = 0.
            green = x / 0.25
            blue = 1.
        return red, green, blue
    def case5():
        # (1.0 red -> 0.0 blue)  smooth across 0 -> 1
        # (1,0,0) -> (0.5,0,0.5) -> (0,0,1)
        #   red   ->    purple   ->  blue
        red = x
        green = 0.
        blue = 1. - x
        return red, green, blue
    def case6():
        # (1.0 red -> 0.5 black -> 0.0 blue)
        if x >= 0.5:
            red = (x - 1. / 2.) / (1. / 2.)
            green = 0.
            blue = 0.
        else:
            red = 0.
            green = 0.
            blue = (1. / 2. - x) / (1. / 2.)
        return red, green, blue
    def case7():
        # (1,0,0) -> (0.5,0.5,0.5) -> (0,0,1)
        #   red   ->    grey       ->  blue
        red = x
        blue = 1. - x
        green = 1. - x if x >= 0.5 else x
        return red, green, blue
    
    switch = {1:case1, 2:case2, 3:case3, 4:case4, 5:case5, 6:case6, 7:case7}
    func = switch[mod]
    return func()
