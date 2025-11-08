import os

"""
Ray Distributed Debugger VSCode Extension (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Starting with Ray 2.39, Anyscale has introduced the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Follow the extension’s installation instructions, then add your cluster using the dashboard URL you obtained earlier.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray Distributed Debugger VSCode extension screenshot

2. Prerequisites.

   Ensure the following are installed (see the extension README for more detail):

   - Visual Studio Code
   - `ray[default]` >= 2.9.1
   - `debugpy` >= 1.8.0

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/c7098b755ff689859837773a916c857.png?raw=true
      :alt: VSCode with Ray prerequisites

3. Environment Variables.

   To enable post‑mortem debugging, set:

   .. code-block:: bash

      export RAY_DEBUG_POST_MORTEM=1

   .. admonition:: Note
      :class: important

      Be sure to remove any legacy flags before starting Ray:

      - `RAY_DEBUG=legacy`
      - `--ray-debugger-external`

4. Configuring BreakpointsSet up breakpoint() in your code, and submit job to cluster. Then the extension will show the breakpoint information.


   1. Insert `breakpoint()` calls into your remote functions.
   2. Submit your job to the cluster.

   The extension will detect active breakpoints and display them in VSCode.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Detected breakpoint in VSCode

   **Note:** Breakpoints are only supported inside functions decorated with `@ray.remote`.

5. Launching the Debugger.

   Run your job directly from the command line (do not use a `launch.json`):

   .. code-block:: bash

      python job.py

6. Attaching to a Breakpoint.

 Once the process hits the first `breakpoint()`, click the Ray Distributed Debugger icon in the VSCode sidebar to attach the debugger.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Attaching VSCode debugger to Ray process

7. Debugging With Multiple breakpoint().

   For each subsequent task, first disconnect the current debugger session, then click the extension icon again to attach to the next breakpoint.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/6e83c910a62c82fecb89c6619e001cd.png?raw=true
      :alt: Disconnecting and reconnecting the debugger
"""

def vscode_conditional_breakpoint(tag=None, rank=-1, once=True):
   """
   根据给定的标签和等级条件，在VSCode中设置条件断点。

   该函数用于在调试时根据特定条件触发断点。如果满足以下条件，则触发断点：
   1. `rank` 参数为 0，或者rank环境变量为0。
   2. 环境变量 `RAY_DEBUG_POST_MORTEM` 已设置。
   3. 如果提供了 `tag` 参数，且该标签存在于环境变量 `DEBUG_TAGS` 中。

   参数:
   - tag (str, 可选): 用于匹配环境变量 `DEBUG_TAGS` 中的标签。如果为None，则无条件触发断点。
   - rank (int, 可选): GPU序号，world rank。

   环境变量:
   - RAY_DEBUG_POST_MORTEM: 如果未设置，函数将直接返回，不触发断点。
   - DEBUG_TAGS: 包含多个标签的字符串，标签之间用 `|` 分隔。如果 `tag` 参数存在于该环境变量中，则触发断点。

   """
   env_tag = f'HIT_BREAKPOINT_REC_{tag}'
   # if rank < 0: rank = os.getenv("RANK", 0)
   # if rank != 0: return
   if not os.getenv('RAY_DEBUG_POST_MORTEM'): return
   if tag is None:
      if once:
         if os.getenv(env_tag, "") != "1":
            os.environ[env_tag] = "1"
            breakpoint()
            return
      else:
         breakpoint()
         return
   else:
      debug_tags = os.getenv('DEBUG_TAGS', '').split('|')
      if tag in debug_tags:
         if once:
            if os.getenv(env_tag, "") != "1":
               os.environ[env_tag] = "1"
               breakpoint()
               return
         else:
            breakpoint()
            return

import pickle

def objdump(obj, file="objdump.tmp"):
   with open(file, "wb+") as f:
      pickle.dump(obj, f)
   return

def objload(file="objdump.tmp"):
   import os
   if not os.path.exists(file):
      return
   with open(file, "rb") as f:
      return pickle.load(f)

bp = vscode_conditional_breakpoint