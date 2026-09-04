from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field


class CpuInfo(BaseModel):
    name: str = "unknown"
    cores: int = 0
    threads: int = 0


class MemoryInfo(BaseModel):
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0


class GpuInfo(BaseModel):
    vendor: str = "unknown"
    name: str = "unknown"
    vram_gb: float = 0.0
    cuda: bool = False
    metal: bool = False
    rocm: bool = False


class DiskInfo(BaseModel):
    free_gb: float = 0.0


class ToolsInfo(BaseModel):
    ollama: bool = False
    git: bool = False
    python: bool = True
    pip: bool = False
    cmake: bool = False


class SystemProfile(BaseModel):
    os: str = "unknown"
    os_version: str = "unknown"
    arch: str = "unknown"
    cpu: CpuInfo = Field(default_factory=CpuInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    gpu: GpuInfo = Field(default_factory=GpuInfo)
    disk: DiskInfo = Field(default_factory=DiskInfo)
    tools: ToolsInfo = Field(default_factory=ToolsInfo)
    python_version: str = "unknown"

    @property
    def ram_gb(self) -> float:
        return self.memory.ram_total_gb

    @property
    def vram_gb(self) -> float:
        return self.gpu.vram_gb

    @property
    def is_apple_silicon(self) -> bool:
        return self.os == "macOS" and self.arch == "arm64"

    @property
    def has_dedicated_gpu(self) -> bool:
        return self.gpu.vendor.lower() in ("nvidia", "amd") or (
            self.is_apple_silicon and self.memory.ram_total_gb >= 8
        )


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _cpu_info() -> CpuInfo:
    name = platform.processor() or "unknown"
    cores = 0
    threads = 0
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 0
        threads = psutil.cpu_count(logical=True) or 0
    except ImportError:
        import os

        threads = os.cpu_count() or 0
        cores = threads // 2 or threads
    return CpuInfo(name=name, cores=cores, threads=threads)


def _memory_info() -> MemoryInfo:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return MemoryInfo(
            ram_total_gb=round(vm.total / 1024**3, 1),
            ram_available_gb=round(vm.available / 1024**3, 1),
        )
    except ImportError:
        pass
    # macOS fallback: sysctl
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "hw.memsize"], text=True, timeout=3)
            total_bytes = int(out.split(":")[1].strip())
            total_gb = round(total_bytes / 1024**3, 1)
            return MemoryInfo(ram_total_gb=total_gb, ram_available_gb=total_gb * 0.5)
        except Exception:
            pass
    # Linux fallback: /proc/meminfo
    try:
        mem = Path("/proc/meminfo").read_text()
        total = next(int(ln.split()[1]) for ln in mem.splitlines() if ln.startswith("MemTotal"))
        avail = next(int(ln.split()[1]) for ln in mem.splitlines() if ln.startswith("MemAvailable"))
        return MemoryInfo(
            ram_total_gb=round(total / 1024**2, 1), ram_available_gb=round(avail / 1024**2, 1)
        )
    except Exception:
        return MemoryInfo()


def _gpu_info() -> GpuInfo:
    system = platform.system()

    # Apple Silicon — unified memory, Metal
    if system == "Darwin" and platform.machine() == "arm64":
        vram = 0.0
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"], text=True, timeout=5
            )
            for line in out.splitlines():
                if "Memory:" in line:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == "Memory:" and i + 2 < len(parts):
                            vram = float(parts[i + 1])  # shared
            import psutil

            vram = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:
            pass
        return GpuInfo(vendor="Apple", name="Apple Silicon (Metal)", vram_gb=vram, metal=True)

    # NVIDIA CUDA
    cuda = False
    vram_gb = 0.0
    gpu_name = "unknown"
    vendor = "unknown"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0]
        gpu_name, mem_mb = line.split(",")
        gpu_name = gpu_name.strip()
        vram_gb = round(float(mem_mb.strip()) / 1024, 1)
        vendor = "NVIDIA"
        cuda = True
    except Exception:
        pass

    # AMD ROCm
    rocm = False
    if vendor == "unknown":
        try:
            out = subprocess.check_output(["rocminfo"], text=True, timeout=5)
            if "gfx" in out.lower():
                vendor = "AMD"
                rocm = True
                gpu_name = "AMD GPU"
        except Exception:
            pass

    # Intel / integrated (fallback)
    if vendor == "unknown":
        try:
            if system == "Linux":
                out = subprocess.check_output(["lspci"], text=True, timeout=5)
                for line in out.splitlines():
                    if "VGA" in line or "Display" in line:
                        gpu_name = line.split(":")[-1].strip()[:60]
                        if "Intel" in gpu_name:
                            vendor = "Intel"
                        elif "AMD" in gpu_name:
                            vendor = "AMD"
                        break
            elif system == "Windows":
                out = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    text=True,
                    timeout=5,
                )
                all_lines = out.splitlines()
                lines = [ln.strip() for ln in all_lines if ln.strip() and ln.strip() != "Name"]
                if lines:
                    gpu_name = lines[0]
                    if "Intel" in gpu_name:
                        vendor = "Intel"
                    elif "NVIDIA" in gpu_name:
                        vendor = "NVIDIA"
                        cuda = _safe(lambda: bool(shutil.which("nvidia-smi")), False)
                    elif "AMD" in gpu_name or "Radeon" in gpu_name:
                        vendor = "AMD"
        except Exception:
            pass

    return GpuInfo(vendor=vendor, name=gpu_name, vram_gb=vram_gb, cuda=cuda, rocm=rocm, metal=False)


def _disk_info() -> DiskInfo:
    try:
        import psutil

        usage = psutil.disk_usage("/")
        return DiskInfo(free_gb=round(usage.free / 1024**3, 1))
    except Exception:
        try:
            stat = shutil.disk_usage("/")
            return DiskInfo(free_gb=round(stat.free / 1024**3, 1))
        except Exception:
            return DiskInfo()


def _tools_info() -> ToolsInfo:
    def has(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    return ToolsInfo(
        ollama=has("ollama"),
        git=has("git"),
        python=True,
        pip=has("pip") or has("pip3"),
        cmake=has("cmake"),
    )


def collect() -> SystemProfile:
    """Sistemin hardware/software profilini topla."""
    system = platform.system()
    os_name = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(system, system)

    return SystemProfile(
        os=os_name,
        os_version=platform.version()[:40],
        arch=platform.machine(),
        cpu=_cpu_info(),
        memory=_memory_info(),
        gpu=_gpu_info(),
        disk=_disk_info(),
        tools=_tools_info(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
