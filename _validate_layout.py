#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用驱动搭真实 e2b_train 三窗口布局 + dummy 命令, 验证编排修好 (不启重训练)."""
import os
import time
from tmux_driver import TmuxDriver

drv = TmuxDriver()
drv._tmux(["kill-session", "-t", "e2b_train"], check=False)

spec = {
    "e2b_train": {
        "fwd":     ["source /tmp/e2b_env_new.sh",
                    "cd /mnt/data_cpfs/qingxu.fu/agentjet/tutorial/e2b_atbench",
                    "echo FWD_READY && touch /tmp/proof_fwd"],
        "server":  ["source /tmp/e2b_env_new.sh",
                    "cd /mnt/data_cpfs/qingxu.fu/agentjet",
                    "echo SERVER_READY && touch /tmp/proof_server"],
        "trainer": ["source /tmp/e2b_env_new.sh",
                    "cd /mnt/data_cpfs/qingxu.fu/agentjet",
                    "echo TRAINER_READY && touch /tmp/proof_trainer"],
    }
}
drv.run(spec)
time.sleep(1.2)

print("=== proof 文件落地 (命令在正确 window 执行的决定性证据) ===")
for w in ["fwd", "server", "trainer"]:
    p = "/tmp/proof_" + w
    print("  " + w + ": " + ("LANDED_OK" if os.path.exists(p) else "MISSING"))
print("=== windows ===", drv.list_windows("e2b_train"))
drv.show_status("e2b_train")
print("\n=== attach 查看: tmux attach -t e2b_train  (Ctrl-b 0/1/2 切窗) ===")
