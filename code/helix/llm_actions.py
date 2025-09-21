"""Strict JSON action API with comprehensive safety guards for LLM environments.

This module provides a type-safe, validated action system that prevents malicious
or malformed inputs while enabling rich LLM interaction capabilities.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ActionType(Enum):
    """Enumeration of allowed action types."""
    ADVANCE_DEPTH = "advance_depth"
    TRAIN_EPOCHS = "train_epochs"
    RECOMPUTE_DIAGNOSTICS = "recompute_diagnostics"
    TOGGLE_ULAM = "toggle_ulam"
    COMPUTE_K_THEORY = "compute_k_theory"
    COMPUTE_PERSISTENT_HOMOLOGY = "compute_ph"
    PLOT_REGIONS = "plot_regions"
    PLOT_CP_MAP = "plot_cp"
    PLOT_ULAM = "plot_ulam"
    SAVE_REPORT = "save_report"
    GET_TIMELINE = "get_timeline"
    SET_PARAMETER = "set_parameter"


@dataclass(frozen=True)
class ActionSchema:
    """Schema definition for action validation."""
    action_type: ActionType
    required_params: List[str]
    optional_params: Dict[str, Any]
    param_types: Dict[str, type]
    param_constraints: Dict[str, Dict[str, Any]]


# Define action schemas with strict validation rules
ACTION_SCHEMAS = {
    ActionType.ADVANCE_DEPTH: ActionSchema(
        action_type=ActionType.ADVANCE_DEPTH,
        required_params=[],
        optional_params={"steps": 1},
        param_types={"steps": int},
        param_constraints={"steps": {"min": 1, "max": 10}}
    ),

    ActionType.TRAIN_EPOCHS: ActionSchema(
        action_type=ActionType.TRAIN_EPOCHS,
        required_params=["n"],
        optional_params={"learning_rate": None, "batch_size": None},
        param_types={"n": int, "learning_rate": float, "batch_size": int},
        param_constraints={
            "n": {"min": 1, "max": 1000},
            "learning_rate": {"min": 1e-6, "max": 1.0},
            "batch_size": {"min": 1, "max": 10000}
        }
    ),

    ActionType.RECOMPUTE_DIAGNOSTICS: ActionSchema(
        action_type=ActionType.RECOMPUTE_DIAGNOSTICS,
        required_params=[],
        optional_params={"tolerance": 1e-10, "include_ph": True, "include_capacity": True},
        param_types={"tolerance": float, "include_ph": bool, "include_capacity": bool},
        param_constraints={"tolerance": {"min": 1e-15, "max": 1e-3}}
    ),

    ActionType.TOGGLE_ULAM: ActionSchema(
        action_type=ActionType.TOGGLE_ULAM,
        required_params=["enabled"],
        optional_params={"bins": 25, "samples_per_cell": 4},
        param_types={"enabled": bool, "bins": int, "samples_per_cell": int},
        param_constraints={
            "bins": {"min": 5, "max": 100},
            "samples_per_cell": {"min": 1, "max": 16}
        }
    ),

    ActionType.COMPUTE_K_THEORY: ActionSchema(
        action_type=ActionType.COMPUTE_K_THEORY,
        required_params=[],
        optional_params={"method": "hodge", "tolerance": 1e-10},
        param_types={"method": str, "tolerance": float},
        param_constraints={
            "method": {"allowed": ["hodge", "smith", "spectral"]},
            "tolerance": {"min": 1e-15, "max": 1e-3}
        }
    ),

    ActionType.COMPUTE_PERSISTENT_HOMOLOGY: ActionSchema(
        action_type=ActionType.COMPUTE_PERSISTENT_HOMOLOGY,
        required_params=[],
        optional_params={"maxdim": 2, "sample_cap": 1024},
        param_types={"maxdim": int, "sample_cap": int},
        param_constraints={
            "maxdim": {"min": 0, "max": 3},
            "sample_cap": {"min": 10, "max": 5000}
        }
    ),

    ActionType.PLOT_REGIONS: ActionSchema(
        action_type=ActionType.PLOT_REGIONS,
        required_params=[],
        optional_params={"show": False, "save_path": None, "dpi": 300},
        param_types={"show": bool, "save_path": str, "dpi": int},
        param_constraints={
            "dpi": {"min": 72, "max": 600},
            "save_path": {"max_length": 200, "allowed_extensions": [".png", ".pdf", ".svg"]}
        }
    ),

    ActionType.SAVE_REPORT: ActionSchema(
        action_type=ActionType.SAVE_REPORT,
        required_params=["path"],
        optional_params={"format": "json", "include_plots": False},
        param_types={"path": str, "format": str, "include_plots": bool},
        param_constraints={
            "path": {"max_length": 200, "no_traversal": True},
            "format": {"allowed": ["json", "yaml", "html"]}
        }
    ),

    ActionType.GET_TIMELINE: ActionSchema(
        action_type=ActionType.GET_TIMELINE,
        required_params=[],
        optional_params={"window": 10, "compress": True},
        param_types={"window": int, "compress": bool},
        param_constraints={"window": {"min": 1, "max": 100}}
    ),

    ActionType.SET_PARAMETER: ActionSchema(
        action_type=ActionType.SET_PARAMETER,
        required_params=["name", "value"],
        optional_params={},
        param_types={"name": str, "value": Union[int, float, bool, str]},
        param_constraints={
            "name": {"allowed": [
                "mass_weight", "wasted_weight", "entropy_weight",
                "cp_weight", "gap_weight", "enable_logging"
            ]},
            "value": {"type_dependent": True}
        }
    ),
}


@dataclass(frozen=True)
class ValidatedAction:
    """Validated action with guaranteed safe parameters."""
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: float
    source: str  # "llm", "user", "system"
    validation_passed: bool = True


class ActionValidationError(ValueError):
    """Raised when action validation fails."""
    pass


class ActionSecurityError(Exception):
    """Raised when action poses security risk."""
    pass


class SafeActionValidator:
    """Comprehensive validator for LLM actions with security checks."""

    def __init__(
        self,
        *,
        allowed_base_paths: Optional[List[str]] = None,
        max_memory_mb: int = 1000,
        max_execution_time_seconds: int = 300,
        enable_path_traversal_check: bool = True,
        enable_resource_limits: bool = True
    ):
        self.allowed_base_paths = allowed_base_paths or ["/tmp", "./outputs"]
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time_seconds
        self.enable_path_check = enable_path_traversal_check
        self.enable_resource_limits = enable_resource_limits

        # Rate limiting
        self._action_history: List[float] = []
        self._max_actions_per_minute = 60

    def validate_action(
        self,
        action_data: Union[str, Dict[str, Any]],
        source: str = "llm"
    ) -> ValidatedAction:
        """Validate and sanitize an action with comprehensive safety checks.

        Parameters
        ----------
        action_data : Union[str, Dict[str, Any]]
            Raw action data (JSON string or dictionary)
        source : str
            Source of the action ("llm", "user", "system")

        Returns
        -------
        ValidatedAction
            Validated and sanitized action

        Raises
        ------
        ActionValidationError
            If action validation fails
        ActionSecurityError
            If action poses security risk
        """
        timestamp = time.time()

        # Rate limiting
        self._check_rate_limit(timestamp)

        # Parse action data
        if isinstance(action_data, str):
            try:
                parsed_data = json.loads(action_data)
            except json.JSONDecodeError as e:
                raise ActionValidationError(f"Invalid JSON: {e}")
        else:
            parsed_data = action_data

        if not isinstance(parsed_data, dict):
            raise ActionValidationError("Action must be a JSON object")

        # Extract action type
        action_type_str = parsed_data.get("action_type") or parsed_data.get("tool")
        if not action_type_str:
            raise ActionValidationError("Missing required field: action_type")

        # Validate action type
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            allowed_types = [t.value for t in ActionType]
            raise ActionValidationError(
                f"Unknown action type '{action_type_str}'. "
                f"Allowed types: {allowed_types}"
            )

        # Get schema
        schema = ACTION_SCHEMAS[action_type]

        # Extract and validate parameters
        args = parsed_data.get("args", {})
        if not isinstance(args, dict):
            raise ActionValidationError("args must be a dictionary")

        validated_params = self._validate_parameters(args, schema)

        # Security checks
        self._perform_security_checks(action_type, validated_params, source)

        return ValidatedAction(
            action_type=action_type,
            parameters=validated_params,
            timestamp=timestamp,
            source=source,
            validation_passed=True
        )

    def _check_rate_limit(self, timestamp: float) -> None:
        """Check if action rate limit is exceeded."""
        # Clean old entries
        cutoff = timestamp - 60.0  # 1 minute window
        self._action_history = [t for t in self._action_history if t > cutoff]

        # Check limit
        if len(self._action_history) >= self._max_actions_per_minute:
            raise ActionSecurityError("Rate limit exceeded: too many actions per minute")

        self._action_history.append(timestamp)

    def _validate_parameters(
        self,
        params: Dict[str, Any],
        schema: ActionSchema
    ) -> Dict[str, Any]:
        """Validate parameters against schema."""
        validated = {}

        # Check required parameters
        for req_param in schema.required_params:
            if req_param not in params:
                raise ActionValidationError(f"Missing required parameter: {req_param}")

        # Validate all parameters
        all_params = {**schema.optional_params, **params}
        for param_name, param_value in all_params.items():
            if param_name in params:  # Use provided value
                param_value = params[param_name]
            elif param_name in schema.optional_params:  # Use default
                param_value = schema.optional_params[param_name]
            else:
                # Unknown parameter - strict mode rejects it
                raise ActionValidationError(f"Unknown parameter: {param_name}")

            # Type validation
            if param_name in schema.param_types:
                expected_type = schema.param_types[param_name]
                if not self._check_type(param_value, expected_type):
                    raise ActionValidationError(
                        f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                        f"got {type(param_value).__name__}"
                    )

            # Constraint validation
            if param_name in schema.param_constraints:
                self._validate_constraints(param_name, param_value, schema.param_constraints[param_name])

            validated[param_name] = param_value

        return validated

    def _check_type(self, value: Any, expected_type: type) -> bool:
        """Check if value matches expected type."""
        if expected_type == Union[int, float, bool, str]:
            return isinstance(value, (int, float, bool, str))
        return isinstance(value, expected_type)

    def _validate_constraints(
        self,
        param_name: str,
        value: Any,
        constraints: Dict[str, Any]
    ) -> None:
        """Validate parameter constraints."""
        if "min" in constraints and value < constraints["min"]:
            raise ActionValidationError(
                f"Parameter '{param_name}' value {value} below minimum {constraints['min']}"
            )

        if "max" in constraints and value > constraints["max"]:
            raise ActionValidationError(
                f"Parameter '{param_name}' value {value} above maximum {constraints['max']}"
            )

        if "allowed" in constraints and value not in constraints["allowed"]:
            raise ActionValidationError(
                f"Parameter '{param_name}' value '{value}' not in allowed values: {constraints['allowed']}"
            )

        if "max_length" in constraints and isinstance(value, str) and len(value) > constraints["max_length"]:
            raise ActionValidationError(
                f"Parameter '{param_name}' exceeds maximum length {constraints['max_length']}"
            )

        if "allowed_extensions" in constraints and isinstance(value, str):
            if not any(value.lower().endswith(ext.lower()) for ext in constraints["allowed_extensions"]):
                raise ActionValidationError(
                    f"Parameter '{param_name}' must have one of these extensions: {constraints['allowed_extensions']}"
                )

        if "no_traversal" in constraints and constraints["no_traversal"] and isinstance(value, str):
            if self.enable_path_check:
                self._check_path_traversal(value)

    def _perform_security_checks(
        self,
        action_type: ActionType,
        params: Dict[str, Any],
        source: str
    ) -> None:
        """Perform comprehensive security checks."""
        # Path traversal checks for file operations
        if action_type in [ActionType.SAVE_REPORT, ActionType.PLOT_REGIONS]:
            path_param = params.get("path") or params.get("save_path")
            if path_param and self.enable_path_check:
                self._check_path_traversal(path_param)
                self._check_allowed_base_paths(path_param)

        # Resource limit checks
        if self.enable_resource_limits:
            if action_type == ActionType.TRAIN_EPOCHS:
                epochs = params.get("n", 0)
                if epochs > 100 and source == "llm":
                    raise ActionSecurityError("LLM-initiated training limited to 100 epochs")

            if action_type == ActionType.COMPUTE_PERSISTENT_HOMOLOGY:
                sample_cap = params.get("sample_cap", 1024)
                if sample_cap > 2000 and source == "llm":
                    raise ActionSecurityError("LLM-initiated PH computation limited to 2000 samples")

        # Memory-intensive operation checks
        memory_intensive_actions = [
            ActionType.COMPUTE_PERSISTENT_HOMOLOGY,
            ActionType.PLOT_REGIONS,
            ActionType.COMPUTE_K_THEORY
        ]
        if action_type in memory_intensive_actions and source == "llm":
            # Could add runtime memory monitoring here
            pass

    def _check_path_traversal(self, path: str) -> None:
        """Check for path traversal attempts."""
        # Normalize path to detect traversal attempts
        try:
            normalized = Path(path).resolve()
            path_str = str(normalized)
        except Exception:
            raise ActionSecurityError(f"Invalid path: {path}")

        # Check for suspicious patterns
        dangerous_patterns = ["../", "..\\", "~", "$"]
        for pattern in dangerous_patterns:
            if pattern in path:
                raise ActionSecurityError(f"Path traversal attempt detected: {path}")

        # Check for absolute paths outside allowed areas
        if normalized.is_absolute():
            allowed = any(
                str(normalized).startswith(base_path)
                for base_path in self.allowed_base_paths
            )
            if not allowed:
                raise ActionSecurityError(f"Path outside allowed directories: {path}")

    def _check_allowed_base_paths(self, path: str) -> None:
        """Ensure path is within allowed base directories."""
        try:
            resolved_path = Path(path).resolve()
        except Exception:
            raise ActionSecurityError(f"Cannot resolve path: {path}")

        # Check if path starts with any allowed base path
        allowed = any(
            str(resolved_path).startswith(Path(base).resolve().as_posix())
            for base in self.allowed_base_paths
        )

        if not allowed:
            raise ActionSecurityError(
                f"Path '{path}' not within allowed directories: {self.allowed_base_paths}"
            )


def create_production_validator() -> SafeActionValidator:
    """Create validator with production security settings."""
    return SafeActionValidator(
        allowed_base_paths=["/tmp/helix", "./outputs", "./reports"],
        max_memory_mb=2000,
        max_execution_time_seconds=600,
        enable_path_traversal_check=True,
        enable_resource_limits=True
    )


def create_development_validator() -> SafeActionValidator:
    """Create validator with relaxed settings for development."""
    return SafeActionValidator(
        allowed_base_paths=["/tmp", "./", "../"],
        max_memory_mb=8000,
        max_execution_time_seconds=1800,
        enable_path_traversal_check=False,
        enable_resource_limits=False
    )


class ActionExecutor:
    """Safe executor for validated actions."""

    def __init__(self, validator: SafeActionValidator):
        self.validator = validator
        self._execution_history: List[ValidatedAction] = []

    def execute_action(
        self,
        action_data: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        source: str = "llm"
    ) -> Dict[str, Any]:
        """Execute a validated action safely.

        Parameters
        ----------
        action_data : Union[str, Dict[str, Any]]
            Raw action data
        context : Optional[Dict[str, Any]]
            Execution context (environment state, etc.)
        source : str
            Source of the action

        Returns
        -------
        Dict[str, Any]
            Execution result
        """
        # Validate action
        try:
            validated_action = self.validator.validate_action(action_data, source)
        except (ActionValidationError, ActionSecurityError) as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }

        # Execute action
        try:
            result = self._execute_validated_action(validated_action, context)
            self._execution_history.append(validated_action)
            return {
                "success": True,
                "result": result,
                "action": validated_action.action_type.value,
                "timestamp": validated_action.timestamp
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "action": validated_action.action_type.value,
                "timestamp": validated_action.timestamp
            }

    def _execute_validated_action(
        self,
        action: ValidatedAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a validated action."""
        # This would integrate with the actual environment
        # For now, return a mock response based on action type

        if action.action_type == ActionType.ADVANCE_DEPTH:
            return {"depth_advanced": action.parameters.get("steps", 1)}

        elif action.action_type == ActionType.TRAIN_EPOCHS:
            return {
                "epochs_trained": action.parameters["n"],
                "final_loss": 0.1  # Mock value
            }

        elif action.action_type == ActionType.GET_TIMELINE:
            window = action.parameters.get("window", 10)
            return {
                "timeline": self._get_mock_timeline(window),
                "compressed": action.parameters.get("compress", True)
            }

        # Add more action implementations as needed
        return {"action_executed": action.action_type.value}

    def _get_mock_timeline(self, window: int) -> List[Dict[str, Any]]:
        """Get mock timeline data."""
        return [
            {"step": i, "timestamp": time.time() - i * 10, "action": "mock"}
            for i in range(min(window, len(self._execution_history)))
        ]


__all__ = [
    "ActionType",
    "ActionSchema",
    "ValidatedAction",
    "ActionValidationError",
    "ActionSecurityError",
    "SafeActionValidator",
    "ActionExecutor",
    "create_production_validator",
    "create_development_validator",
]