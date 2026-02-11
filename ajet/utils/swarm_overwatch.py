"""
Swarm Overwatch - Real-time monitoring interface for rollout pool information
Similar to htop/nvidia-smi for AgentJet swarm servers
"""

import time
import sys
from datetime import datetime
from typing import Optional

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from loguru import logger

from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import (
    CurrentBatchRolloutPoolInformation,
)


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

    def create_header(self) -> Panel:
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
        table = Table(
            title="Rollout Pool Summary",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            expand=True,
        )

        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Current", justify="right", style="green", width=15)
        table.add_column("Target", justify="right", style="yellow", width=15)
        table.add_column("Progress", justify="right", style="blue", width=15)
        table.add_column("Bar", width=30)

        # Episodes
        ep_cur, ep_tgt, ep_pct = self.create_progress_bar(
            info.completed_episodes, info.completed_episode_target, "Episodes"
        )
        ep_bar = self._create_text_bar(ep_pct)
        table.add_row(
            "Episodes", f"{ep_cur:,}", f"{ep_tgt:,}", f"{ep_pct:.1f}%", ep_bar
        )

        # Tasks
        task_cur, task_tgt, task_pct = self.create_progress_bar(
            info.completed_tasks, info.completed_task_target, "Tasks"
        )
        task_bar = self._create_text_bar(task_pct)
        table.add_row(
            "Tasks", f"{task_cur:,}", f"{task_tgt:,}", f"{task_pct:.1f}%", task_bar
        )

        # Non-dummy tasks
        nd_cur, nd_tgt, nd_pct = self.create_progress_bar(
            info.completed_non_dummy_tasks,
            info.completed_non_dummy_task_target,
            "Non-Dummy Tasks",
        )
        nd_bar = self._create_text_bar(nd_pct)
        table.add_row(
            "Non-Dummy Tasks", f"{nd_cur:,}", f"{nd_tgt:,}", f"{nd_pct:.1f}%", nd_bar
        )

        # Expected repeats
        table.add_row(
            "Expected Repeats", "-", f"{info.task_expected_num_repeat:,}", "-", "-"
        )

        return table

    def _create_text_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based progress bar"""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{'green' if percentage >= 100 else 'yellow'}]{bar}[/]"

    def create_task_details_table(
        self, info: CurrentBatchRolloutPoolInformation
    ) -> Table:
        """Create detailed task completion table"""
        table = Table(
            title="Task Completion Details",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            expand=True,
        )

        table.add_column("Task ID", style="cyan", no_wrap=True, width=40)
        table.add_column("Episodes", justify="right", style="green", width=15)
        table.add_column("Episode UUIDs (first 3)", style="dim", overflow="fold")

        if not info.completed_tasks_details:
            table.add_row("[dim]No task details available[/dim]", "", "")
            return table

        # Sort tasks by number of completed episodes (descending)
        sorted_tasks = sorted(
            info.completed_tasks_details.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Limit to top 15 tasks to fit in terminal
        for task_id, episode_uuids in sorted_tasks[:15]:
            # Show first 3 episode UUIDs
            preview_uuids = episode_uuids[:3]
            uuid_str = ", ".join([f"{uuid[:8]}..." for uuid in preview_uuids])
            if len(episode_uuids) > 3:
                uuid_str += f" (+{len(episode_uuids) - 3} more)"

            table.add_row(
                task_id[:40] if len(task_id) > 40 else task_id,
                f"{len(episode_uuids):,}",
                uuid_str,
            )

        if len(sorted_tasks) > 15:
            table.add_row(
                f"[dim]... and {len(sorted_tasks) - 15} more tasks[/dim]", "", ""
            )

        return table

    def create_dashboard(
        self, info: Optional[CurrentBatchRolloutPoolInformation]
    ) -> Layout:
        """Create the main dashboard layout"""
        layout = Layout()

        # Create header
        header = self.create_header()

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
            # Show data
            summary = self.create_summary_table(info)
            details = self.create_task_details_table(info)

            layout.split_column(
                Layout(header, size=8), Layout(summary, size=12), Layout(details)
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
