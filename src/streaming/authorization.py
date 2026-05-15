"""
Authorization Module

Implements authorization and access control for HistoCore Real-Time WSI Streaming.

Requirements: Role-based access control, permission management
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class Role:
    """User role with permissions."""

    def __init__(self, name: str, permissions: Set[str]):
        self.name = name
        self.permissions = permissions

    def has_permission(self, permission: str) -> bool:
        """Check if role has permission."""
        return permission in self.permissions


class AuthorizationManager:
    """Manages authorization and access control."""

    def __init__(self):
        """Initialize authorization manager."""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}

        # Define default roles
        self._initialize_default_roles()

        logger.info("Authorization manager initialized")

    def _initialize_default_roles(self):
        """Initialize default roles."""
        # Admin role - full access
        self.roles["admin"] = Role(
            "admin", {"read", "write", "delete", "admin", "manage_users", "manage_roles"}
        )

        # User role - read/write access
        self.roles["user"] = Role("user", {"read", "write"})

        # Viewer role - read-only access
        self.roles["viewer"] = Role("viewer", {"read"})

    def add_role(self, role: Role):
        """Add custom role."""
        self.roles[role.name] = role
        logger.info(f"Added role: {role.name}")

    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user."""
        if role_name not in self.roles:
            raise ValueError(f"Role not found: {role_name}")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")

    def revoke_role(self, user_id: str, role_name: str):
        """Revoke role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role {role_name} from user {user_id}")

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission."""
        if user_id not in self.user_roles:
            return False

        for role_name in self.user_roles[user_id]:
            role = self.roles.get(role_name)
            if role and role.has_permission(permission):
                return True

        return False

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user."""
        permissions = set()

        if user_id in self.user_roles:
            for role_name in self.user_roles[user_id]:
                role = self.roles.get(role_name)
                if role:
                    permissions.update(role.permissions)

        return permissions
