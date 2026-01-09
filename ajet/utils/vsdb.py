import os
import pickle

"""
Ray Distributed Debugger VSCode Extension (Recommended)

Starting with Ray 2.39, Anyscale has introduced the
`Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension.

1. Install the Ray Distributed Debugger extension in VSCode.

2. In AgentJet project:

   2-1. In the place your want to set a conditional breakpoint, write
         `from ajet import bp; bp("TAG_1")`

   2-2. When launching the training process, add `--debug` argument
         `ajet --conf your_config.yaml --debug="TAG_1"`

   2-3. Open Tab "Ray Distributed Debugger" in VSCode, and just wait until the breakpoint is hit.

"""

def vscode_conditional_breakpoint(tag=None, once=True):

   env_tag = f'HIT_BREAKPOINT_REC_{tag}'
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
