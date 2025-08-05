#!/usr/bin/env python3
"""
OpenHands Docker Disk Space Manager

A monitoring and cleanup tool for OpenHands Docker containers and images.
Monitors disk usage and automatically cleans up stopped OpenHands containers
and unused OpenHands images to prevent disk space issues on shared systems.

Features:
- Configurable monitoring intervals
- Disk usage thresholds
- OpenHands-specific container/image identification
- Safe cleanup (only targets OpenHands resources)
- Comprehensive logging
- Graceful error handling
"""

import argparse
import json
import logging
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import docker
from docker.models.containers import Container
from docker.models.images import Image


@dataclass
class CleanupConfig:
    """Configuration for the disk space manager."""

    # Monitoring intervals
    check_interval_seconds: int = 300  # 5 minutes
    cleanup_interval_seconds: int = 1800  # 30 minutes

    # Disk usage thresholds
    disk_usage_warning_threshold: float = 80.0  # 80%
    disk_usage_critical_threshold: float = 90.0  # 90%

    # Container cleanup settings
    cleanup_stopped_containers: bool = True
    container_age_threshold_minutes: int = 10

    # Image cleanup settings
    cleanup_unused_images: bool = True
    image_age_threshold_minutes: int = 10
    keep_latest_images: int = 3  # Always keep the 3 most recent OpenHands images

    # Safety settings
    dry_run: bool = False
    max_containers_per_cleanup: int = 50
    max_images_per_cleanup: int = 20

    # Paths
    docker_root_path: str = "/var/lib/docker"
    log_file: Optional[str] = None


class OpenHandsIdentifier:
    """Identifies OpenHands containers and images based on naming patterns."""

    # OpenHands image patterns from the documentation
    OPENHANDS_IMAGE_PATTERNS = [
        r'^oh_v\d+\.\d+\.\d+_.*',  # Versioned tag: oh_v{version}_{details}
        r'^ghcr\.io/all-hands-ai/openhands.*',  # GitHub registry
        r'^.*openhands.*',  # General OpenHands pattern
    ]

    # OpenHands container name patterns
    OPENHANDS_CONTAINER_PATTERNS = [
        r'.*openhands.*',
        r'.*oh_v\d+.*',
        r'.*all-hands-ai.*',
    ]

    @classmethod
    def is_openhands_image(cls, image: Image) -> bool:
        """Check if an image is related to OpenHands."""
        try:
            # Check image tags
            for tag in image.tags:
                if cls._matches_patterns(tag.lower(), cls.OPENHANDS_IMAGE_PATTERNS):
                    return True

            # Check image repository/name in attributes
            if hasattr(image, 'attrs') and 'RepoTags' in image.attrs:
                for repo_tag in image.attrs['RepoTags'] or []:
                    if cls._matches_patterns(repo_tag.lower(), cls.OPENHANDS_IMAGE_PATTERNS):
                        return True

            return False
        except Exception as e:
            logging.warning(f"Error checking image {image.id}: {e}")
            return False

    @classmethod
    def is_openhands_container(cls, container: Container) -> bool:
        """Check if a container is related to OpenHands."""
        try:
            # Check container name
            if container.name and cls._matches_patterns(container.name.lower(), cls.OPENHANDS_CONTAINER_PATTERNS):
                return True

            # Check image name
            if hasattr(container, 'image') and container.image:
                image_name = str(container.image).lower()
                if cls._matches_patterns(image_name, cls.OPENHANDS_IMAGE_PATTERNS):
                    return True

            # Check labels for OpenHands markers
            if hasattr(container, 'labels') and container.labels:
                for label_key, label_value in container.labels.items():
                    if 'openhands' in label_key.lower() or 'openhands' in str(label_value).lower():
                        return True

            return False
        except Exception as e:
            logging.warning(f"Error checking container {container.id}: {e}")
            return False

    @staticmethod
    def _matches_patterns(text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given regex patterns."""
        return any(re.match(pattern, text) for pattern in patterns)


class DiskUsageMonitor:
    """Monitors disk usage and provides usage statistics."""

    def __init__(self, docker_root_path: str = "/var/lib/docker"):
        self.docker_root_path = docker_root_path

    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics."""
        try:
            # Get overall disk usage for the Docker root directory
            usage = shutil.disk_usage(self.docker_root_path)
            total = usage.total
            used = usage.used
            free = usage.free

            usage_percent = (used / total) * 100 if total > 0 else 0

            return {
                'total_bytes': total,
                'used_bytes': used,
                'free_bytes': free,
                'usage_percent': usage_percent,
                'free_percent': 100 - usage_percent
            }
        except Exception as e:
            logging.error(f"Error getting disk usage: {e}")
            return {
                'total_bytes': 0,
                'used_bytes': 0,
                'free_bytes': 0,
                'usage_percent': 0,
                'free_percent': 100
            }

    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"


class OpenHandsCleanupManager:
    """Manages cleanup of OpenHands containers and images."""

    def __init__(self, config: CleanupConfig):
        self.config = config
        self.docker_client = docker.from_env()
        self.identifier = OpenHandsIdentifier()
        self.disk_monitor = DiskUsageMonitor(config.docker_root_path)
        self._setup_logging()
        self._running = True

    def _setup_logging(self):
        """Set up logging configuration."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = logging.INFO

        if self.config.log_file:
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.config.log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(level=log_level, format=log_format)

    def get_openhands_containers(self, only_stopped: bool = True) -> List[Container]:
        """Get OpenHands containers, optionally filtering to only stopped ones."""
        try:
            all_containers = self.docker_client.containers.list(all=True)
            openhands_containers = [
                container for container in all_containers
                if self.identifier.is_openhands_container(container)
            ]

            if only_stopped:
                openhands_containers = [
                    container for container in openhands_containers
                    if container.status != 'running'
                ]

            return openhands_containers
        except Exception as e:
            logging.error(f"Error getting OpenHands containers: {e}")
            return []

    def get_openhands_images(self) -> List[Image]:
        """Get OpenHands images."""
        try:
            all_images = self.docker_client.images.list(all=True)
            return [
                image for image in all_images
                if self.identifier.is_openhands_image(image)
            ]
        except Exception as e:
            logging.error(f"Error getting OpenHands images: {e}")
            return []

    def cleanup_stopped_containers(self) -> Dict[str, int]:
        """Clean up stopped OpenHands containers based on age threshold."""
        if not self.config.cleanup_stopped_containers:
            return {'removed': 0, 'errors': 0}

        logging.info("Starting cleanup of stopped OpenHands containers...")

        stopped_containers = self.get_openhands_containers(only_stopped=True)
        age_threshold = timedelta(minutes=self.config.container_age_threshold_minutes)
        current_time = datetime.now()

        removed_count = 0
        error_count = 0

        for container in stopped_containers[:self.config.max_containers_per_cleanup]:
            try:
                # Get container finished time
                finished_at = container.attrs.get('State', {}).get('FinishedAt')
                if not finished_at or finished_at == '0001-01-01T00:00:00Z':
                    continue

                # Parse the timestamp (Docker uses RFC3339 format)
                finished_time = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
                container_age = current_time - finished_time.replace(tzinfo=None)

                if container_age > age_threshold:
                    if self.config.dry_run:
                        logging.info(f"[DRY RUN] Would remove container: {container.name} ({container.id[:12]})")
                    else:
                        container.remove()
                        logging.info(f"Removed container: {container.name} ({container.id[:12]})")
                    removed_count += 1

            except Exception as e:
                logging.error(f"Error removing container {container.id[:12]}: {e}")
                error_count += 1

        logging.info(f"Container cleanup completed: {removed_count} removed, {error_count} errors")
        return {'removed': removed_count, 'errors': error_count}

    def cleanup_unused_images(self) -> Dict[str, int]:
        """Clean up unused OpenHands images, keeping the most recent ones."""
        if not self.config.cleanup_unused_images:
            return {'removed': 0, 'errors': 0}

        logging.info("Starting cleanup of unused OpenHands images...")

        openhands_images = self.get_openhands_images()

        # Sort images by creation date (newest first)
        sorted_images = sorted(
            openhands_images,
            key=lambda img: img.attrs.get('Created', ''),
            reverse=True
        )

        age_threshold = timedelta(minutes=self.config.image_age_threshold_minutes)
        current_time = datetime.now()

        removed_count = 0
        error_count = 0

        for image in sorted_images[self.config.keep_latest_images:]:
            if removed_count >= self.config.max_images_per_cleanup:
                break

            try:
                # Skip if image is in use by containers
                if self._is_image_in_use(image):
                    continue

                # Check image age
                created_str = image.attrs.get('Created', '')
                if created_str:
                    created_time = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    image_age = current_time - created_time.replace(tzinfo=None)

                    if image_age > age_threshold:
                        if self.config.dry_run:
                            tags = image.tags or [image.id[:12]]
                            logging.info(f"[DRY RUN] Would remove image: {tags[0]}")
                        else:
                            image.remove(force=True)
                            tags = image.tags or [image.id[:12]]
                            logging.info(f"Removed image: {tags[0]}")
                        removed_count += 1

            except Exception as e:
                logging.error(f"Error removing image {image.id[:12]}: {e}")
                error_count += 1

        logging.info(f"Image cleanup completed: {removed_count} removed, {error_count} errors")
        return {'removed': removed_count, 'errors': error_count}

    def _is_image_in_use(self, image: Image) -> bool:
        """Check if an image is currently in use by any container."""
        try:
            all_containers = self.docker_client.containers.list(all=True)
            image_id = image.id

            for container in all_containers:
                if container.image.id == image_id:
                    return True
            return False
        except Exception:
            return True  # Err on the side of caution

    def monitor_and_cleanup(self):
        """Main monitoring and cleanup loop."""
        logging.info("Starting OpenHands Docker Disk Space Manager")
        logging.info(f"Check interval: {self.config.check_interval_seconds}s")
        logging.info(f"Cleanup interval: {self.config.cleanup_interval_seconds}s")
        logging.info(f"Dry run mode: {self.config.dry_run}")

        last_cleanup_time = datetime.now() - timedelta(seconds=self.config.cleanup_interval_seconds)

        while self._running:
            try:
                # Monitor disk usage
                disk_usage = self.disk_monitor.get_disk_usage()
                usage_percent = disk_usage['usage_percent']

                logging.info(
                    f"Disk usage: {usage_percent:.1f}% "
                    f"({self.disk_monitor.format_bytes(disk_usage['used_bytes'])}/"
                    f"{self.disk_monitor.format_bytes(disk_usage['total_bytes'])})"
                )

                current_time = datetime.now()
                should_cleanup = False
                cleanup_reason = ""

                # Check if cleanup is needed based on disk usage
                if usage_percent >= self.config.disk_usage_critical_threshold:
                    should_cleanup = True
                    cleanup_reason = f"Critical disk usage: {usage_percent:.1f}%"
                else:
                    # Check if it's time for scheduled cleanup
                    time_since_cleanup = current_time - last_cleanup_time
                    if time_since_cleanup.total_seconds() >= self.config.cleanup_interval_seconds:
                        should_cleanup = True
                        cleanup_reason = f"Scheduled cleanup (usage: {usage_percent:.1f}%)"

                # Perform cleanup if needed
                if should_cleanup:
                    logging.info(f"Starting cleanup{" [DRY RUN]" if self.config.dry_run else ""}: {cleanup_reason}")

                    container_results = self.cleanup_stopped_containers()
                    image_results = self.cleanup_unused_images()

                    logging.info(
                        f"Cleanup completed - "
                        f"Containers: {container_results['removed']} removed, "
                        f"Images: {image_results['removed']} removed"
                    )

                    last_cleanup_time = current_time

                    # Check disk usage after cleanup
                    post_cleanup_usage = self.disk_monitor.get_disk_usage()
                    logging.info(
                        f"Post-cleanup disk usage: {post_cleanup_usage['usage_percent']:.1f}%"
                    )

                # Wait for next check (with responsive shutdown checking)
                self._interruptible_sleep(self.config.check_interval_seconds)

            except KeyboardInterrupt:
                logging.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                self._interruptible_sleep(self.config.check_interval_seconds)

        logging.info("OpenHands Docker Disk Space Manager stopped")

    def _interruptible_sleep(self, duration_seconds: int):
        """Sleep for the specified duration while checking for shutdown signal."""
        sleep_interval = 1.0  # Check for shutdown every second
        elapsed = 0.0

        while elapsed < duration_seconds and self._running:
            remaining = duration_seconds - elapsed
            sleep_time = min(sleep_interval, remaining)
            time.sleep(sleep_time)
            elapsed += sleep_time

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False


def load_config_from_file(config_path: str) -> CleanupConfig:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return CleanupConfig(**config_data)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return CleanupConfig()


def create_sample_config(config_path: str):
    """Create a sample configuration file."""
    config = CleanupConfig()
    config_dict = {
        'check_interval_seconds': config.check_interval_seconds,
        'cleanup_interval_seconds': config.cleanup_interval_seconds,
        'disk_usage_warning_threshold': config.disk_usage_warning_threshold,
        'disk_usage_critical_threshold': config.disk_usage_critical_threshold,
        'cleanup_stopped_containers': config.cleanup_stopped_containers,
        'container_age_threshold_minutes': config.container_age_threshold_minutes,
        'cleanup_unused_images': config.cleanup_unused_images,
        'image_age_threshold_minutes': config.image_age_threshold_minutes,
        'keep_latest_images': config.keep_latest_images,
        'dry_run': config.dry_run,
        'max_containers_per_cleanup': config.max_containers_per_cleanup,
        'max_images_per_cleanup': config.max_images_per_cleanup,
        'docker_root_path': config.docker_root_path,
        'log_file': config.log_file
    }

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Sample configuration created at: {config_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenHands Docker Disk Space Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python docker-disk-space-manager.py

  # Run in dry-run mode
  python docker-disk-space-manager.py --dry-run

  # Use custom configuration file
  python docker-disk-space-manager.py --config config.json

  # Create sample configuration file
  python docker-disk-space-manager.py --create-config config.json

  # Run once and exit (no continuous monitoring)
  python docker-disk-space-manager.py --run-once
        """
    )

    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (JSON format)'
    )
    parser.add_argument(
        '--create-config',
        help='Create a sample configuration file at the specified path'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be cleaned up without actually removing anything'
    )
    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run cleanup once and exit (no continuous monitoring)'
    )
    parser.add_argument(
        '--log-file', '-l',
        help='Log file path'
    )

    args = parser.parse_args()

    # Handle config file creation
    if args.create_config:
        create_sample_config(args.create_config)
        return

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = CleanupConfig()

    # Override config with command line arguments
    if args.dry_run:
        config.dry_run = True
    if args.log_file:
        config.log_file = args.log_file

    # Create and run the cleanup manager
    manager = OpenHandsCleanupManager(config)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logging.info("Received shutdown signal")
        manager.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.run_once:
            # Run cleanup once and exit
            logging.info("Running one-time cleanup...")
            manager.cleanup_stopped_containers()
            manager.cleanup_unused_images()
            logging.info("One-time cleanup completed")
        else:
            # Start continuous monitoring
            manager.monitor_and_cleanup()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
