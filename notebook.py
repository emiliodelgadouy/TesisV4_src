import importlib
import subprocess
import sys
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# GPU: deben quedar fijadas antes del primer import de TensorFlow.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")

from pathlib import Path


class NotebookRuntime:
    """Arranque del notebook: requirements, Comet antes de TF, mixed precision y autoreload."""

    @staticmethod
    def is_running_in_colab() -> bool:
        try:
            import google.colab  # type: ignore
            return True
        except ImportError:
            return False

    @staticmethod
    def install_requirements_silent(requirements_path: str = "requirements.txt") -> None:
        req_path = Path(requirements_path).resolve()

        if not req_path.exists():
            raise FileNotFoundError(f"No existe {req_path}")

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "-r",
            str(req_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Falló la instalación de requirements.\n\n"
                f"Comando:\n{' '.join(cmd)}\n\n"
                f"Error de pip:\n{result.stderr}"
            )

    @staticmethod
    def _ensure_imp_compat() -> None:
        """IPython autoreload still imports `imp`, removed in Python 3.12."""
        if "imp" in sys.modules:
            return

        import types

        imp = types.ModuleType("imp")
        imp.reload = importlib.reload
        sys.modules["imp"] = imp

    @classmethod
    def _enable_autoreload(cls, shell) -> None:
        cls._ensure_imp_compat()
        if "autoreload" not in shell.extension_manager.loaded:
            shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")

    @staticmethod
    def _install_comet_if_missing() -> None:
        if importlib.util.find_spec("comet_ml") is not None:
            return

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "comet_ml",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Falló la instalación de comet_ml.\n\n"
                f"Comando:\n{' '.join(cmd)}\n\n"
                f"Error de pip:\n{result.stderr}"
            )

    @staticmethod
    def _ensure_comet() -> None:
        """Comet ML debe importarse antes que TensorFlow para instrumentar Keras."""
        import comet_ml  # noqa: F401

    @classmethod
    def configure(
        cls,
        autoreload: bool = True,
        disable_bytecode: bool = True,
        requirements_path: str = "src/requirements.txt",
        install_requirements: bool = True,
        skip_requirements_in_colab: bool = True,
        mixed_precision_policy: str | None = "mixed_float16",
        install_comet: bool = True,
    ) -> None:
        running_in_colab = cls.is_running_in_colab()
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "2"
        if disable_bytecode:
            sys.dont_write_bytecode = True

        # Instalar antes de importar Comet/TensorFlow: actualizar el entorno despues
        # de cargar TensorFlow deja esta sesion usando las versiones anteriores.
        if install_requirements and not (running_in_colab and skip_requirements_in_colab):
            cls.install_requirements_silent(requirements_path)
        importlib.invalidate_caches()

        if install_comet:
            cls._install_comet_if_missing()
        cls._ensure_comet()

        if mixed_precision_policy is not None:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy(mixed_precision_policy)

        if not autoreload:
            return

        try:
            from IPython import get_ipython
        except ImportError:
            return

        shell = get_ipython()
        if shell is None:
            return

        try:
            cls._enable_autoreload(shell)
        except Exception as exc:
            print(f"Warning: autoreload disabled ({exc})")


def is_running_in_colab() -> bool:
    return NotebookRuntime.is_running_in_colab()


def configure_notebook(*args, **kwargs) -> None:
    NotebookRuntime.configure(*args, **kwargs)
