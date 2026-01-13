# def patch_task_creation():
#     pass


import inspect
import asyncio
from functools import wraps

def patch_task_creation():
    # Hook asyncio.create_task
    original_create_task = asyncio.create_task

    @wraps(original_create_task)
    def debug_create_task(coro, name=None, **kwargs):
        if not name:
            caller = inspect.stack()[1]
            coro_name = getattr(coro, '__name__', str(coro))
            name = f"DEBUG_{coro_name}_at_{caller.filename.split('/')[-1]}:{caller.lineno}"
            print(f"🎯 asyncio.create_task: {name}")
        return original_create_task(coro, name=name, **kwargs)

    # Hook loop.create_task
    original_loop_create_task = asyncio.AbstractEventLoop.create_task

    def debug_loop_create_task(self, coro, name=None, context=None):
        if not name:
            caller = inspect.stack()[1]
            coro_name = getattr(coro, '__name__', str(coro))
            name = f"DEBUG_{coro_name}_at_{caller.filename.split('/')[-1]}:{caller.lineno}"
            print(f"🎯 loop.create_task: {name}")
        return original_loop_create_task(self, coro, name=name, context=context)

    # Hook Task.__init__
    original_task_init = asyncio.Task.__init__

    def debug_task_init(self, coro, loop=None, name=None, context=None):
        if not name:
            caller = inspect.stack()[2]  # 需要往上找更远的调用者
            coro_name = getattr(coro, '__name__', str(coro))
            name = f"DEBUG_{coro_name}_at_{caller.filename.split('/')[-1]}:{caller.lineno}"
            print(f"🎯 Task.__init__: {name}")
        return original_task_init(self, coro, loop=loop, name=name, context=context)

    asyncio.create_task = debug_create_task
    asyncio.AbstractEventLoop.create_task = debug_loop_create_task

patch_task_creation()