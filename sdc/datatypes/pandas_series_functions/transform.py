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
        sig = func.get_call_type(cpu_target.typing_context, func_args, {})
        last_Arg = sig.args[-1]
        if isinstance(last_Arg, types.StarArgTuple):
            func_args.extend(args)
            sig = func.get_call_type(cpu_target.typing_context, func_args, {})
            output_type = sig.return_type
            # find if final arg of function is *args
            def impl(self, func, *args):
                input_arr = self._data
                length = len(input_arr)

                output_arr = numpy.empty(length, dtype=output_type)

                for i in prange(length):
                    output_arr[i] = func(input_arr[i], *args)

                return pandas.Series(output_arr, index=self._index, name=self._name)
        else:
            output_type = sig.return_type

            def impl(self, func, *args):
                input_arr = self._data
                length = len(input_arr)

                output_arr = numpy.empty(length, dtype=output_type)

                for i in prange(length):
                    output_arr[i] = func(input_arr[i])

                return pandas.Series(output_arr, index=self._index, name=self._name)

        return impl
    elif isinstance(self, SeriesType) and isinstance(func, types.Tuple):
        output_types = []
        output_cols = []
        n_series = len(func)

        for i in prange(n_series):
            sig = func[i].get_call_type(cpu_target.typing_context, func_args, {})
            output_types.append(sig.return_type)
            output_cols.append(func[i].dispatcher.py_func.__name__)

        func_lines = [f"def impl(self, func, *args):"]

        results = []
        for i in range(n_series):
            result_c = f"s_{i}"
            func_lines += [f"  {result_c} = self.transform(func[{i}], *args)"]
            results.append((output_cols[i], result_c))

        data = ', '.join(f'"{col}": {data}' for col, data in results)
        func_lines += [f"  return pandas.DataFrame({{{data}}}, self._index)"]
        func_text = '\n'.join(func_lines)
        print(func_text)
        global_vars = {'pandas': pandas, 'numpy': numpy, 'types': types, 'prange': prange}
        loc_vars = {}
        exec(func_text, global_vars, loc_vars)
        _impl = loc_vars['impl']

        return _impl
