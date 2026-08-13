# -*- coding: utf-8 -*-

import os
import time

from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.schema.task import Task, WorkflowOutput

AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
EPISODE_TYPE = os.getenv("EPISODE_TYPE", "train")
DISCARD_TIMEOUT = int(os.getenv("DISCARD_TIMEOUT", "3600"))


def main():
    swarm = SwarmClient(AJET_SWARM_URL)
    episode_uuid, cred = swarm.begin_episode(
        discard_episode_timeout=DISCARD_TIMEOUT,
        episode_type=EPISODE_TYPE,
    )

    fed_note = "WILL be fed to training" if EPISODE_TYPE == "train" else "will NOT be fed to training (train)"

    print("\n================ Manual Episode Ready ================")
    print(f"episode_uuid : {episode_uuid}")
    print(f"base_url     : {cred.base_url}")
    print(f"api_key      : {cred.api_key}")
    print(f"episode_type : {EPISODE_TYPE}  (data {fed_note})")
    print("------------------------------------------------------")
    print(f'export OPENAI_BASE_URL="{cred.base_url}"')
    print(f'export OPENAI_API_KEY="{cred.api_key}"')
    print("======================================================")
    print("\nPoint your agent/CLI at the base_url + api_key above (model name = your")
    print("policy/training model). This terminal holds the episode open; Ctrl-C to abort.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nAborting episode ...")
        swarm.end_episode(task=Task(main_query="r", task_id="123"), episode_uuid=episode_uuid, workflow_output=WorkflowOutput(reward=1,is_success=True,metadata={"note":"manual"}))
        print("Done.")


if __name__ == "__main__":
    main()
