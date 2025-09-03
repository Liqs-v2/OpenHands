#!/usr/bin/env python3
"""
OpenHands Docker Disk Space Manager

A monitoring and cleanup tool for OpenHands Docker containers and images.
Monitors Docker disk usage using the Docker API and automatically cleans up
stopped OpenHands containers and unused OpenHands images when disk usage
exceeds configurable thresholds.

Features:
- Docker-native disk usage monitoring via Docker API
- Configurable absolute disk usage thresholds (bytes)
- Configurable monitoring intervals
- OpenHands-specific container/image identification
- Safe cleanup (only targets OpenHands resources)
- Comprehensive logging
- Graceful error handling
"""

import argparse
import json
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


import docker
from docker.models.containers import Container
from docker.models.images import Image


@dataclass
class CleanupConfig:
    """Configuration for the disk space manager."""

    # Monitoring intervals
    check_interval_seconds: int = 300  # 5 minutes
    cleanup_interval_seconds: int = 1800  # 30 minutes

    # Docker disk usage thresholds (in GB)
    docker_disk_usage_threshold_gb: int = 150  # 150GB cleanup threshold
    docker_disk_usage_warning_gb: int = 130  # 130GB warning threshold

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

    # Logging
    log_file: Optional[str] = None

    # Optional global prunes (outside OpenHands-only scope)
    # When enabled, these will run the equivalent of:
    #   docker image prune -f           (dangling images only)
    #   docker builder prune -f         (build cache)
    prune_dangling_images: bool = False
    prune_builder_cache: bool = False


class OpenHandsIdentifier:
    """Identifies OpenHands containers and images based on naming patterns."""

    # OpenHands image patterns from the documentation
    OPENHANDS_IMAGE_PATTERNS = [
        r'oh_v\d+\.\d+\.\d+_.*',  # Versioned tag: oh_v{version}_{details}
        r'^ghcr\.io/all-hands-ai.*',  # GitHub registry
        r'^.*all-hands-ai.*',  # General OpenHands pattern
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


class DockerDiskUsageMonitor:
    """Monitors Docker-specific disk usage and provides usage statistics."""

    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client

    def get_docker_disk_usage(self) -> Dict[str, int]:
        """Get Docker disk usage statistics using docker.df() API."""
        try:
            # Get Docker disk usage information (similar to 'docker system df --json')
            usage_data = self.docker_client.df()

            # Calculate total Docker disk usage from different components
            container_bytes = self._calculate_containers_usage(usage_data.get('Containers', []))
            image_bytes = usage_data.get('LayersSize', 0)
            volume_bytes = self._calculate_volumes_usage(usage_data.get('Volumes', []))

            total_docker_bytes = container_bytes + image_bytes + volume_bytes

            return {
                'total_docker_bytes': total_docker_bytes,
                'container_bytes': container_bytes,
                'image_bytes': image_bytes,
                'volume_bytes': volume_bytes,
                'layers_size': usage_data.get('LayersSize', 0),
                'build_cache_bytes': usage_data.get('BuildCache', [])  # May be a list or int depending on version
            }
        except Exception as e:
            logging.error(f"Error getting Docker disk usage: {e}")
            return {
                'total_docker_bytes': 0,
                'container_bytes': 0,
                'image_bytes': 0,
                'volume_bytes': 0,
                'layers_size': 0,
                'build_cache_bytes': 0
            }

    def _calculate_containers_usage(self, containers_data: List[Dict]) -> int:
        """Calculate total disk usage from containers."""
        try:
            return sum(
                container.get('SizeRootFs', 0)
                for container in containers_data
                if container.get('SizeRootFs') is not None
            )
        except Exception as e:
            logging.warning(f"Error calculating container usage: {e}")
            return 0

    def _calculate_volumes_usage(self, volumes_data: List[Dict]) -> int:
        """Calculate total disk usage from volumes."""
        try:
            total_volume_bytes = 0
            for volume in volumes_data:
                usage_data = volume.get('UsageData')
                if usage_data and isinstance(usage_data, dict):
                    size = usage_data.get('Size', 0)
                    if isinstance(size, int):
                        total_volume_bytes += size
            return total_volume_bytes
        except Exception as e:
            logging.warning(f"Error calculating volume usage: {e}")
            return 0

    def is_cleanup_threshold_exceeded(self, total_docker_bytes: int, threshold_gb: int) -> bool:
        """Check if Docker disk usage exceeds the cleanup threshold."""
        threshold_bytes = self._gb_to_bytes(threshold_gb)
        return total_docker_bytes >= threshold_bytes

    def is_warning_threshold_exceeded(self, total_docker_bytes: int, warning_threshold_gb: int) -> bool:
        """Check if Docker disk usage exceeds the warning threshold."""
        warning_threshold_bytes = self._gb_to_bytes(warning_threshold_gb)
        return total_docker_bytes >= warning_threshold_bytes

    def _gb_to_bytes(self, gb: int) -> int:
        """Convert GB to bytes."""
        return gb * 1024**3

    def format_gb(self, bytes_value: int) -> str:
        """Format bytes as GB for human-readable display."""
        return f"{bytes_value / (1024**3):.1f} GB"

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
        self.docker_disk_monitor = DockerDiskUsageMonitor(self.docker_client)
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
            key=lambda img: img.attrs.get('Created', '1970-01-01T00:00:00Z'),
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
        logging.info(f"Docker disk usage cleanup threshold: {self.config.docker_disk_usage_threshold_gb} GB")
        logging.info(f"Docker disk usage warning threshold: {self.config.docker_disk_usage_warning_gb} GB")
        logging.info(f"Dry run mode: {self.config.dry_run}")

        last_cleanup_time = datetime.now() - timedelta(seconds=self.config.cleanup_interval_seconds)

        while self._running:
            try:
                # Monitor Docker disk usage
                docker_usage = self.docker_disk_monitor.get_docker_disk_usage()
                total_docker_bytes = docker_usage['total_docker_bytes']

                logging.info(
                    f"Docker disk usage: {self.docker_disk_monitor.format_gb(total_docker_bytes)} "
                    f"(Containers: {self.docker_disk_monitor.format_bytes(docker_usage['container_bytes'])}, "
                    f"Images: {self.docker_disk_monitor.format_bytes(docker_usage['image_bytes'])}, "
                    f"Volumes: {self.docker_disk_monitor.format_bytes(docker_usage['volume_bytes'])})"
                )

                current_time = datetime.now()
                should_cleanup = False
                cleanup_reason = ""

                # Check if cleanup is needed based on Docker disk usage threshold
                if self.docker_disk_monitor.is_cleanup_threshold_exceeded(
                    total_docker_bytes, self.config.docker_disk_usage_threshold_gb
                ):
                    should_cleanup = True
                    cleanup_reason = (
                        f"Docker disk usage exceeded threshold: "
                        f"{self.docker_disk_monitor.format_gb(total_docker_bytes)} >= "
                        f"{self.config.docker_disk_usage_threshold_gb} GB"
                    )
                elif self.docker_disk_monitor.is_warning_threshold_exceeded(
                    total_docker_bytes, self.config.docker_disk_usage_warning_gb
                ):
                    # Check if it's time for scheduled cleanup when approaching threshold
                    time_since_cleanup = current_time - last_cleanup_time
                    if time_since_cleanup.total_seconds() >= self.config.cleanup_interval_seconds:
                        should_cleanup = True
                        cleanup_reason = (
                            f"Scheduled cleanup - approaching threshold: "
                            f"{self.docker_disk_monitor.format_gb(total_docker_bytes)} >= "
                            f"{self.config.docker_disk_usage_warning_gb} GB"
                        )
                else:
                    # Check if it's time for regular scheduled cleanup
                    time_since_cleanup = current_time - last_cleanup_time
                    if time_since_cleanup.total_seconds() >= self.config.cleanup_interval_seconds:
                        should_cleanup = True
                        cleanup_reason = (
                            f"Regular scheduled cleanup "
                            f"(Docker usage: {self.docker_disk_monitor.format_bytes(total_docker_bytes)})"
                        )

                # Perform cleanup if needed
                if should_cleanup:
                    logging.info(f"Starting cleanup{" [DRY RUN]" if self.config.dry_run else ""}: {cleanup_reason}")

                    container_results = self.cleanup_stopped_containers()
                    image_results = self.cleanup_unused_images()

                    # Optional global prunes
                    reclaimed_total = 0
                    if self.config.prune_dangling_images:
                        reclaimed_total += self.prune_dangling_images()
                    if self.config.prune_builder_cache:
                        reclaimed_total += self.prune_build_cache()

                    logging.info(
                        f"Cleanup completed - "
                        f"Containers: {container_results['removed']} removed, "
                        f"Images: {image_results['removed']} removed, "
                        f"Reclaimed: {self.docker_disk_monitor.format_bytes(reclaimed_total)}"
                    )

                    last_cleanup_time = current_time

                    # Check Docker disk usage after cleanup
                    post_cleanup_usage = self.docker_disk_monitor.get_docker_disk_usage()
                    post_cleanup_total = post_cleanup_usage['total_docker_bytes']
                    logging.info(
                        f"Post-cleanup Docker disk usage: {self.docker_disk_monitor.format_gb(post_cleanup_total)}"
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

    # -------------------------
    # Global prune integrations (SDK)
    # -------------------------
    def prune_dangling_images(self) -> int:
        """Prune dangling images using Docker SDK and return bytes reclaimed."""
        if self.config.dry_run:
            logging.info("[DRY RUN] Would prune dangling images (docker images.prune)")
            return 0
        try:
            # Equivalent to: docker image prune -f (dangling only)
            result = self.docker_client.images.prune(filters={'dangling': True})
            reclaimed = int(result.get('SpaceReclaimed', 0) or 0)
            deleted_count = len(result.get('ImagesDeleted', []) or [])
            logging.info(
                "Images prune completed: deleted=%d, reclaimed=%s",
                deleted_count,
                self.docker_disk_monitor.format_bytes(reclaimed),
            )
            return reclaimed
        except docker.errors.APIError as api_err:
            logging.error("Images prune failed: %s", str(api_err))
            return 0
        except Exception as e:
            logging.error("Unexpected error during images prune: %s", str(e))
            return 0

    def prune_build_cache(self) -> int:
        """Prune builder cache using Docker SDK and return bytes reclaimed."""
        if self.config.dry_run:
            logging.info("[DRY RUN] Would prune builder cache (docker api.prune_builds)")
            return 0
        try:
            # Equivalent to: docker builder prune -f (default: unused cache)
            result = self.docker_client.api.prune_builds()
            reclaimed = int(result.get('SpaceReclaimed', 0) or 0)
            logging.info(
                "Builder cache prune completed: reclaimed=%s",
                self.docker_disk_monitor.format_bytes(reclaimed),
            )
            return reclaimed
        except docker.errors.APIError as api_err:
            logging.error("Builder prune failed: %s", str(api_err))
            return 0
        except Exception as e:
            logging.error("Unexpected error during builder prune: %s", str(e))
            return 0


def load_config_from_file(config_path: str) -> CleanupConfig:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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
        'docker_disk_usage_threshold_gb': config.docker_disk_usage_threshold_gb,
        'docker_disk_usage_warning_gb': config.docker_disk_usage_warning_gb,
        'cleanup_stopped_containers': config.cleanup_stopped_containers,
        'container_age_threshold_minutes': config.container_age_threshold_minutes,
        'cleanup_unused_images': config.cleanup_unused_images,
        'image_age_threshold_minutes': config.image_age_threshold_minutes,
        'keep_latest_images': config.keep_latest_images,
        'dry_run': config.dry_run,
        'max_containers_per_cleanup': config.max_containers_per_cleanup,
        'max_images_per_cleanup': config.max_images_per_cleanup,
        'log_file': config.log_file,
        'prune_dangling_images': config.prune_dangling_images,
        'prune_builder_cache': config.prune_builder_cache,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
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
    parser.add_argument(
        '--prune-dangling-images',
        action='store_true',
        help='Run docker image prune -f (dangling images) during cleanup'
    )
    parser.add_argument(
        '--prune-builder-cache',
        action='store_true',
        help='Run docker builder prune -f (build cache) during cleanup'
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
    if getattr(args, 'prune_dangling_images', False):
        config.prune_dangling_images = True
    if getattr(args, 'prune_builder_cache', False):
        config.prune_builder_cache = True

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
            if manager.config.prune_dangling_images:
                manager.prune_dangling_images()
            if manager.config.prune_builder_cache:
                manager.prune_build_cache()
            logging.info("One-time cleanup completed")
        else:
            # Start continuous monitoring
            manager.monitor_and_cleanup()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
