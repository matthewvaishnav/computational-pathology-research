"""
WebSocket handler for real-time collaboration
Manages connections and broadcasts updates to multiple pathologists
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import WebSocket


class CollaborationManager:
    """
    Manages real-time collaboration between multiple pathologists
    viewing the same slide
    """

    def __init__(self):
        # slide_id -> set of WebSocket connections
        self.connections: Dict[str, Set[WebSocket]] = {}

        # slide_id -> user_id -> cursor position
        self.cursors: Dict[str, Dict[str, dict]] = {}

        # slide_id -> list of recent actions
        self.action_history: Dict[str, List[dict]] = {}

        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, slide_id: str, user_id: str):
        """
        Connect a user to a slide collaboration session

        Args:
            websocket: WebSocket connection
            slide_id: Slide identifier
            user_id: User/expert identifier
        """
        await websocket.accept()

        # Add to connections
        if slide_id not in self.connections:
            self.connections[slide_id] = set()
            self.cursors[slide_id] = {}
            self.action_history[slide_id] = []

        self.connections[slide_id].add(websocket)

        # Send current state to new user
        await self._send_current_state(websocket, slide_id, user_id)

        # Notify others of new user
        await self.broadcast(
            slide_id,
            {"type": "user_joined", "user_id": user_id, "timestamp": datetime.now().isoformat()},
            exclude=websocket,
        )

        self.logger.info(f"User {user_id} connected to slide {slide_id}")

    def disconnect(self, websocket: WebSocket, slide_id: str, user_id: str):
        """
        Disconnect a user from a slide collaboration session

        Args:
            websocket: WebSocket connection
            slide_id: Slide identifier
            user_id: User/expert identifier
        """
        if slide_id in self.connections:
            self.connections[slide_id].discard(websocket)

            # Remove cursor
            if slide_id in self.cursors and user_id in self.cursors[slide_id]:
                del self.cursors[slide_id][user_id]

            # Clean up if no more connections
            if not self.connections[slide_id]:
                del self.connections[slide_id]
                if slide_id in self.cursors:
                    del self.cursors[slide_id]
                if slide_id in self.action_history:
                    del self.action_history[slide_id]

        self.logger.info(f"User {user_id} disconnected from slide {slide_id}")

    async def broadcast(self, slide_id: str, message: dict, exclude: Optional[WebSocket] = None):
        """
        Broadcast message to all users viewing a slide

        Args:
            slide_id: Slide identifier
            message: Message to broadcast
            exclude: Optional WebSocket to exclude from broadcast
        """
        if slide_id not in self.connections:
            return

        disconnected = set()

        for connection in self.connections[slide_id]:
            if connection == exclude:
                continue

            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.connections[slide_id].discard(conn)

    async def handle_cursor_move(self, slide_id: str, user_id: str, cursor_data: dict):
        """
        Handle cursor movement from a user

        Args:
            slide_id: Slide identifier
            user_id: User identifier
            cursor_data: Cursor position and metadata
        """
        if slide_id not in self.cursors:
            self.cursors[slide_id] = {}

        self.cursors[slide_id][user_id] = {**cursor_data, "timestamp": datetime.now().isoformat()}

        # Broadcast cursor position to others
        await self.broadcast(
            slide_id, {"type": "cursor_move", "user_id": user_id, "cursor": cursor_data}
        )

    async def handle_annotation_action(self, slide_id: str, user_id: str, action: dict):
        """
        Handle annotation action (create, update, delete)

        Args:
            slide_id: Slide identifier
            user_id: User identifier
            action: Action data
        """
        # Add to history
        if slide_id not in self.action_history:
            self.action_history[slide_id] = []

        action_record = {**action, "user_id": user_id, "timestamp": datetime.now().isoformat()}

        self.action_history[slide_id].append(action_record)

        # Keep only last 100 actions
        if len(self.action_history[slide_id]) > 100:
            self.action_history[slide_id] = self.action_history[slide_id][-100:]

        # Broadcast action to others
        await self.broadcast(slide_id, {"type": "annotation_action", "action": action_record})

    async def _send_current_state(self, websocket: WebSocket, slide_id: str, user_id: str):
        """
        Send current collaboration state to newly connected user

        Args:
            websocket: WebSocket connection
            slide_id: Slide identifier
            user_id: User identifier
        """
        # Get active users (excluding self)
        active_users = []
        if slide_id in self.cursors:
            active_users = [uid for uid in self.cursors[slide_id].keys() if uid != user_id]

        # Get recent actions
        recent_actions = []
        if slide_id in self.action_history:
            recent_actions = self.action_history[slide_id][-10:]  # Last 10 actions

        # Send state
        await websocket.send_json(
            {
                "type": "initial_state",
                "active_users": active_users,
                "cursors": self.cursors.get(slide_id, {}),
                "recent_actions": recent_actions,
            }
        )

    def get_active_users(self, slide_id: str) -> List[str]:
        """Get list of active users for a slide"""
        if slide_id in self.cursors:
            return list(self.cursors[slide_id].keys())
        return []

    def get_connection_count(self, slide_id: str) -> int:
        """Get number of active connections for a slide"""
        if slide_id in self.connections:
            return len(self.connections[slide_id])
        return 0


# Global collaboration manager instance
collaboration_manager = CollaborationManager()
