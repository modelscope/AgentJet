"""
Swarm Overwatch - Real-time monitoring interface for rollout pool information
Similar to htop/nvidia-smi for AgentJet swarm servers
"""

import time
from datetime import datetime
from typing import Optional

import httpx
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from loguru import logger

from ajet.tuner_lib.experimental.swarm_overwatch_utils import CurrentBatchRolloutPoolInformation


class SwarmOverwatch:
    """Real-time monitoring interface for swarm rollout pool"""

    def __init__(self, server_url: str, refresh_interval: float = 1.0):
        """
        Initialize the overwatch monitor

        Args:
            server_url: Base URL of the swarm server (e.g., http://localhost:10086)
            refresh_interval: Refresh interval in seconds (default: 1.0)
        """
        self.server_url = server_url.rstrip("/")
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.last_update_time = None
        self.error_count = 0
        self.total_requests = 0

    def fetch_pool_info(self) -> Optional[CurrentBatchRolloutPoolInformation]:
        """Fetch current batch rollout pool information from server"""
        try:
            response = httpx.get(
                f"{self.server_url}/get_current_batch_rollout_pool_information",
                timeout=5.0,
            )
            response.raise_for_status()
            self.total_requests += 1
            self.last_update_time = datetime.now()
            data = CurrentBatchRolloutPoolInformation.model_validate(response.json())
            return data
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to fetch pool info: {e}")
            return None

    def create_header(
        self, info: Optional[CurrentBatchRolloutPoolInformation] = None
    ) -> Panel:
        """Create header panel with server info"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_update = (
            self.last_update_time.strftime("%H:%M:%S")
            if self.last_update_time
            else "Never"
        )

        header_text = Text()
        header_text.append("AgentJet Swarm Overwatch", style="bold cyan")
        header_text.append(f"\nServer: {self.server_url}", style="dim")
        header_text.append(f"\nCurrent Time: {now}", style="green")
        header_text.append(f"  |  Last Update: {last_update}", style="yellow")
        header_text.append(f"  |  Refresh: {self.refresh_interval}s", style="blue")
        header_text.append(f"\nRequests: {self.total_requests}", style="magenta")
        header_text.append(
            f"  |  Errors: {self.error_count}",
            style="red" if self.error_count > 0 else "green",
        )

        # Add engine status and global step if available
        if info:
            if info.engine_status:
                header_text.append(
                    f"\nEngine Status: {info.engine_status}", style="bold yellow"
                )
                # Show booting time if engine is booting
                if info.engine_status == "ENGINE.BOOTING" and info.booting_start_time is not None:
                    booting_duration = int(time.time() - info.booting_start_time)
                    header_text.append(
                        f"  |  Booting Time: {booting_duration}s", style="bold cyan"
                    )
            if info.global_step is not None:
                header_text.append(
                    f"  |  Global Step (Model's Weight Version): {info.global_step:,}", style="bold blue"
                )

        return Panel(header_text, border_style="bright_blue", padding=(0, 1))

    def create_progress_bar(self, current: int, target: int, title: str) -> tuple:
        """Create a progress bar representation"""
        if target == 0:
            percentage = 0.0
        else:
            percentage = (current / target) * 100

        return current, target, percentage

    def create_summary_table(self, info: CurrentBatchRolloutPoolInformation) -> Table:
        """Create summary statistics table"""
        # Determine border style based on engine status
        is_active = info.engine_status in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]
        border_style = "blue" if is_active else "dim"
        title_prefix = "" if is_active else "[WAITING ENGINE.ROLLING] "

        table = Table(
            title=f"{title_prefix}Completed Episode Pool Summary (Progress to Hit Next Weight Update)",
            show_header=True,
            header_style="bold magenta",
            border_style=border_style,
            expand=True,
        )

        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Current", justify="right", style="green", width=15)
        table.add_column("Target", justify="right", style="yellow", width=15)
        table.add_column("Progress", justify="right", style="blue", width=15)
        table.add_column("Bar", width=30)

        # Determine which row to highlight based on sample_collection_method
        highlight_episodes = (
            info.sample_collection_method == "rollout_until_finish_enough_episodes"
        )
        highlight_tasks = (
            info.sample_collection_method == "rollout_until_finish_enough_tasks"
        )
        highlight_non_dummy = (
            info.sample_collection_method
            == "rollout_until_finish_enough_non_dummy_tasks"
        )

        # Episodes
        ep_cur, ep_tgt, ep_pct = self.create_progress_bar(
            info.completed_episodes, info.completed_episode_target, "Completed Episodes"
        )
        ep_bar = self._create_text_bar(ep_pct)
        ep_metric = (
            "-> *Completed Episodes (chosen)*"
            if highlight_episodes
            else "Completed Episodes"
        )
        ep_style = "bold green" if highlight_episodes else None
        table.add_row(
            f"[{ep_style}]{ep_metric}[/{ep_style}]"
            if highlight_episodes
            else ep_metric,
            f"{ep_cur:,}",
            f"{ep_tgt:,}",
            f"{ep_pct:.1f}%",
            ep_bar,
            style=ep_style if highlight_episodes else None,
        )

        # Tasks
        task_cur, task_tgt, task_pct = self.create_progress_bar(
            info.completed_tasks, info.completed_task_target, "Completed Tasks"
        )
        task_bar = self._create_text_bar(task_pct)
        task_metric = (
            "-> *Completed Tasks (chosen)*" if highlight_tasks else "Completed Tasks"
        )
        task_style = "bold green" if highlight_tasks else None
        table.add_row(
            f"[{task_style}]{task_metric}[/{task_style}]"
            if highlight_tasks
            else task_metric,
            f"{task_cur:,}",
            f"{task_tgt:,}",
            f"{task_pct:.1f}%",
            task_bar,
            style=task_style if highlight_tasks else None,
        )

        # Non-dummy tasks
        nd_cur, nd_tgt, nd_pct = self.create_progress_bar(
            info.completed_non_dummy_tasks,
            info.completed_non_dummy_task_target,
            "Completed Non-Dummy Tasks",
        )
        nd_bar = self._create_text_bar(nd_pct)
        nd_metric = (
            "-> *Completed Non-Dummy Tasks (chosen)*"
            if highlight_non_dummy
            else "Completed Non-Dummy Tasks"
        )
        nd_style = "bold green" if highlight_non_dummy else None
        table.add_row(
            f"[{nd_style}]{nd_metric}[/{nd_style}]"
            if highlight_non_dummy
            else nd_metric,
            f"{nd_cur:,}",
            f"{nd_tgt:,}",
            f"{nd_pct:.1f}%",
            nd_bar,
            style=nd_style if highlight_non_dummy else None,
        )

        # Average episodes per task
        if info.completed_tasks_details:
            episodes_per_task = [len(episode_list) for episode_list in info.completed_tasks_details.values()]
            avg_episodes_per_task = sum(episodes_per_task) / len(episodes_per_task) if episodes_per_task else 0.0
        else:
            avg_episodes_per_task = 0.0
        table.add_row(
            "Average Episode Per Task",
            f"{avg_episodes_per_task:.2f}",
            f"{info.task_expected_num_repeat:,}",
            "-",
            "-"
        )

        return table

    def _create_text_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based progress bar"""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{'green' if percentage >= 100 else 'yellow'}]{bar}[/]"

    def create_running_episodes_table(
        self, info: CurrentBatchRolloutPoolInformation
    ) -> Table:
        """Create running episodes table"""
        # Determine border style based on engine status
        is_active = info.engine_status in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]
        border_style = "blue" if is_active else "dim"
        title_prefix = "" if is_active else "[WAITING ENGINE.ROLLING] "

        # Add total counts to title if running episodes exist
        title = f"{title_prefix}Running Episodes"
        if info.running_episode_details:
            num_episodes = len(info.running_episode_details)
            # Count unique tasks from running episodes
            unique_tasks = set()
            for details in info.running_episode_details.values():
                task_id = details.get("task_id")
                if task_id:
                    unique_tasks.add(task_id)
            num_tasks = len(unique_tasks)
            title = f"{title_prefix}Running Episodes (Episodes: {num_episodes})"

        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            border_style=border_style,
            expand=True,
        )

        table.add_column("Episode UUID", style="cyan", no_wrap=True, width=20, overflow="ellipsis")
        table.add_column("Status", style="green")
        table.add_column("LLM Calls", style="magenta", justify="right")
        table.add_column("Last Req / Patience", style="yellow")

        if not info.running_episode_details:
            table.add_row("[dim]No running episodes[/dim]", "", "", "")
            return table

        # Sort by time since last activity (descending)
        sorted_episodes = sorted(
            info.running_episode_details.items(),
            key=lambda x: float(x[1]["time_since_last_activity"].rstrip("s")),
            reverse=True,
        )

        for episode_uuid, details in sorted_episodes[:30]:
            last_req = details["time_since_last_activity"]
            patience = details.get("discard_episode_timeout", "N/A")
            llm_calls = details.get("llm_call_count", "0")
            table.add_row(
                episode_uuid[:40] if len(episode_uuid) > 40 else episode_uuid,
                details["episode_status"],
                llm_calls,
                f"{last_req} / {patience}",
            )

        if len(sorted_episodes) > 30:
            table.add_row(
                f"[dim]... and {len(sorted_episodes) - 30} more episodes[/dim]", "", "", ""
            )

        return table

    def create_task_details_table(
        self, info: CurrentBatchRolloutPoolInformation
    ) -> Table:
        """Create detailed task completion table"""
        # Determine border style based on engine status
        is_active = info.engine_status in ["ENGINE.ROLLING", "ENGINE.ROLLING_POST"]
        border_style = "blue" if is_active else "dim"
        title_prefix = "" if is_active else "[WAITING ENGINE.ROLLING] "

        table = Table(
            title=f"{title_prefix}Task Completion Details",
            show_header=True,
            header_style="bold magenta",
            border_style=border_style,
            expand=True,
        )

        table.add_column("Task ID", style="cyan", no_wrap=True, overflow="ellipsis")
        table.add_column("Episodes", justify="right", style="green")
        table.add_column("Reward", justify="right", style="yellow")
        table.add_column("Episode UUIDs (first 3)", style="dim", overflow="fold")

        if not info.completed_tasks_details:
            table.add_row("[dim]No task details available[/dim]", "", "", "")
            return table

        # Sort tasks by number of completed episodes (descending)
        sorted_tasks = sorted(
            info.completed_tasks_details.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Limit to top 30 tasks to fit in terminal
        for task_id, episode_uuids in sorted_tasks[:30]:
            # Show first 3 episode UUIDs
            preview_uuids = episode_uuids[:3]
            uuid_str = ", ".join([f"{uuid[:8]}..." for uuid in preview_uuids])
            if len(episode_uuids) > 3:
                uuid_str += f" (+{len(episode_uuids) - 3} more)"

            # Calculate reward statistics
            reward_str = "-"
            if info.completed_tasks_rewards and task_id in info.completed_tasks_rewards:
                rewards = info.completed_tasks_rewards[task_id]
                if rewards:
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    reward_str = f"{mean_reward:.3f} ± {std_reward:.3f}"

            table.add_row(
                task_id[:40] if len(task_id) > 40 else task_id,
                f"{len(episode_uuids):,}",
                reward_str,
                uuid_str,
            )

        if len(sorted_tasks) > 30:
            table.add_row(
                f"[dim]... and {len(sorted_tasks) - 30} more tasks[/dim]", "", "", ""
            )

        return table

    def create_logo_panel(self, info: CurrentBatchRolloutPoolInformation) -> Text:
        """Create logo display for OFFLINE and BOOTING states"""
        logo = """
     █████╗  ██████╗ ███████╗███╗   ██╗████████╗     ██╗███████╗████████╗
    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝     ██║██╔════╝╚══██╔══╝
    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║        ██║█████╗     ██║
    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██   ██║██╔══╝     ██║
    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ╚█████╔╝███████╗   ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚════╝ ╚══════╝   ╚═╝
        """

        content = Text()
        content.append(logo, style="bold cyan")
        content.append("\n\n")

        if info.engine_status == "ENGINE.OFFLINE":
            content.append("Engine Status: ", style="bold white")
            content.append("OFFLINE", style="bold yellow")
            content.append("\n\n")
            content.append("Ready to accept commands:\n", style="bold green")
            content.append(f"  • swarm_client = SwarmClient(\"{self.server_url}\")\n", style="cyan")
            content.append("  • swarm_client.sync_train_config()\n", style="cyan")
            content.append("  • swarm_client.start_engine()\n", style="cyan")
            content.append("\n")
            content.append("Please sync your training configuration and start the engine to begin rollouts.", style="dim")
        elif info.engine_status == "ENGINE.BOOTING":
            content.append("Engine Status: ", style="bold white")
            content.append("BOOTING", style="bold yellow")
            if info.booting_start_time is not None:
                booting_duration = int(time.time() - info.booting_start_time)
                content.append(f"  ({booting_duration}s)", style="bold cyan")
            content.append("\n\n")
            content.append("Engine is booting...\n", style="bold green")
            content.append("Please wait, we are loading model weights and ray workers, this will take a few minutes.\n", style="dim")

        return content

    def create_dashboard(
        self, info: Optional[CurrentBatchRolloutPoolInformation]
    ) -> Layout:
        """Create the main dashboard layout"""
        layout = Layout()

        # Create header
        header = self.create_header(info)

        if info is None:
            # Show error state
            error_panel = Panel(
                "[bold red]Failed to fetch data from server[/bold red]\n"
                f"[dim]Attempted to connect to: {self.server_url}[/dim]\n"
                f"[dim]Total errors: {self.error_count}[/dim]",
                border_style="red",
                padding=(1, 2),
            )
            layout.split_column(Layout(header, size=8), Layout(error_panel))
        else:
            # Check engine status and show logo for OFFLINE or BOOTING states
            if info.engine_status in ["ENGINE.OFFLINE", "ENGINE.BOOTING"]:
                # Hide tables and show logo
                logo_display = self.create_logo_panel(info)
                layout.split_column(
                    Layout(header, size=8),
                    Layout(logo_display),
                )
            else:
                # Show data tables for other engine states
                summary = self.create_summary_table(info)
                running_episodes = self.create_running_episodes_table(info)
                details = self.create_task_details_table(info)

                # Create a horizontal layout for running episodes and task details
                bottom_row = Layout()
                bottom_row.split_row(
                    Layout(running_episodes, name="running"),
                    Layout(details, name="details"),
                )

                layout.split_column(
                    Layout(header, size=8),
                    Layout(summary, size=12),
                    Layout(bottom_row),
                )

        return layout

    def run(self):
        """Start the monitoring interface"""
        self.console.clear()

        try:
            with Live(
                self.create_dashboard(None),
                console=self.console,
                refresh_per_second=1,
                screen=True,
            ) as live:
                self.console.print(
                    "[bold green]Starting Swarm Overwatch...[/bold green]"
                )
                self.console.print(f"[dim]Press Ctrl+C to exit[/dim]\n")
                time.sleep(1)

                while True:
                    try:
                        # Fetch latest data
                        info = self.fetch_pool_info()

                        # Update display
                        live.update(self.create_dashboard(info))

                        # Wait for next refresh
                        time.sleep(self.refresh_interval)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logger.error(f"Error in monitoring loop: {e}")
                        time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            self.console.clear()
            self.console.print("\n[bold yellow]Overwatch stopped by user[/bold yellow]")
            self.console.print(
                f"[dim]Total requests: {self.total_requests}, Errors: {self.error_count}[/dim]\n"
            )


def start_overwatch(server_url: str, refresh_interval: float = 1.0):
    """
    Start the swarm overwatch monitoring interface

    Args:
        server_url: Base URL of the swarm server
        refresh_interval: Refresh interval in seconds (default: 1.0)
    """
    overwatch = SwarmOverwatch(server_url, refresh_interval)
    overwatch.run()


if __name__ == "__main__":
    # Test with default localhost
    start_overwatch("http://localhost:10086")
