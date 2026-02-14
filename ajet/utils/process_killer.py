import os
from loguru import logger

def is_key_episode_status(key: str) -> bool:
    return key.startswith("episodes-")

def is_key_finished_episode_status(key: str) -> bool:
    return key.startswith("finished-episodes-")

def kill_process_tree(shared_mem_dict_lock=None, shared_mem_dict=None):
    logger.exception("[stop_engine] Initiating engine shutdown and cleanup...")
    try:
        import psutil

        killed_pids = []
        errors = []

        # Get the training process PID if it exists
        if shared_mem_dict and shared_mem_dict_lock:
            training_pid = shared_mem_dict.get('training_process_pid', None)
        else:
            training_pid = os.getpid()

        if training_pid is not None:
            try:
                # Try to get the process and all its children
                try:
                    parent = psutil.Process(training_pid)
                    children = parent.children(recursive=True)

                    # Kill all child processes first
                    for child in children:
                        try:
                            logger.info(f"[stop_engine] Terminating child process PID: {child.pid}")
                            child.terminate()
                            killed_pids.append(child.pid)
                        except psutil.NoSuchProcess:
                            logger.warning(f"[stop_engine] Child process {child.pid} already terminated")
                        except Exception as e:
                            logger.error(f"[stop_engine] Error terminating child process {child.pid}: {e}")
                            errors.append(f"Child {child.pid}: {str(e)}")

                    # Wait for children to terminate gracefully
                    gone, alive = psutil.wait_procs(children, timeout=16)

                    # Force kill any remaining children
                    for p in alive:
                        try:
                            logger.warning(f"[stop_engine] Force killing child process PID: {p.pid}")
                            p.kill()
                        except Exception as e:
                            logger.error(f"[stop_engine] Error force killing child {p.pid}: {e}")
                            errors.append(f"Force kill child {p.pid}: {str(e)}")

                    # Now terminate the parent process
                    logger.info(f"[stop_engine] Terminating parent process PID: {training_pid}")
                    parent.terminate()
                    killed_pids.append(training_pid)

                    # Wait for parent to terminate gracefully
                    try:
                        parent.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        logger.warning(f"[stop_engine] Force killing parent process PID: {training_pid}")
                        parent.kill()

                except psutil.NoSuchProcess:
                    logger.warning(f"[stop_engine] Process {training_pid} not found (may have already terminated)")

            except Exception as e:
                logger.error(f"[stop_engine] Error killing training process: {e}")
                errors.append(f"Training process: {str(e)}")
        else:
            logger.info("[stop_engine] No training process PID found in shared memory")

        # Clean up all episodes from shared memory
        episode_keys = []
        if shared_mem_dict and shared_mem_dict_lock:
            with shared_mem_dict_lock:
                episode_keys = [k for k in shared_mem_dict.keys() if is_key_episode_status(k) or is_key_finished_episode_status(k)]
                for key in episode_keys:
                    del shared_mem_dict[key]
                    logger.info(f"[stop_engine] Removed episode: {key}")

                # Clear unclaimed episodes list
                if 'unclaimed_episodes' in shared_mem_dict:
                    num_unclaimed = len(shared_mem_dict['unclaimed_episodes'])
                    shared_mem_dict['unclaimed_episodes'] = []
                    logger.info(f"[stop_engine] Cleared {num_unclaimed} unclaimed episodes")

                # Reset engine status to OFFLINE
                shared_mem_dict['engine_status'] = "ENGINE.OFFLINE"

                # Remove training process PID
                if 'training_process_pid' in shared_mem_dict:
                    del shared_mem_dict['training_process_pid']

                logger.info("[stop_engine] Engine status set to OFFLINE")

        result = {
            "success": True,
            "killed_pids": killed_pids,
            "episodes_removed": len(episode_keys) if 'episode_keys' in locals() else 0,
        }

        if errors:
            result["warnings"] = errors
            logger.warning(f"[stop_engine] Completed with warnings: {errors}")
        else:
            logger.info(f"[stop_engine] Successfully terminated engine and reset state")

        return result

    except Exception as e:
        logger.error(f"[stop_engine] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

        # Even if there's an error, try to reset the status
        try:
            if shared_mem_dict and shared_mem_dict_lock:
                with shared_mem_dict_lock:
                    shared_mem_dict['engine_status'] = "ENGINE.OFFLINE"
        except:
            pass

        return {"success": False, "error": str(e)}
