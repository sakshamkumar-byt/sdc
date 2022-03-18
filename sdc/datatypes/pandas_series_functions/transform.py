from typing import Union

import numpy
import pandas
from numba import types, prange

from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.utilities.utils import sdc_overload_method
from numba.core.registry import cpu_target

@sdc_overload_method(SeriesType, 'transform')
def pd_series_overload_single_func_args(self, func, *args):
    func_args = [self.dtype]
    if isinstance(self, SeriesType) and isinstance(func, types.Callable):
        isinstance(self, types.BaseTuple)

        func_args.extend(args)
        sig = func.get_call_type(cpu_target.typing_context, func_args, {})
        output_type = sig.return_type

        # find if final arg of function is *args
        def impl(self, func, *args):
            print(args.count)
            input_arr = self._data
            length = len(input_arr)

            output_arr = numpy.empty(length, dtype=output_type)

            for i in prange(length):
                output_arr[i] = func(input_arr[i], *args)

            return pandas.Series(output_arr, index=self._index, name=self._name)

        return impl