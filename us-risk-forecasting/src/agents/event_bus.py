"""
Event bus for publish-subscribe communication between agents.
"""
import logging
from typing import Callable, Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import threading


logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event bus for publish-subscribe communication.
    
    Allows agents to subscribe to specific event types and receive
    notifications when those events are published.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the event bus.
        
        Args:
            enable_logging: Whether to log event publications and subscriptions
        """
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
        self._enable_logging = enable_logging
        self._lock = threading.Lock()
        
        if self._enable_logging:
            logger.info("EventBus initialized")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Name of the event type to subscribe to (e.g., 'DataUpdateEvent')
            callback: Function to call when event is published. Should accept event as parameter.
        
        Example:
            event_bus.subscribe('DataUpdateEvent', agent.on_data_update)
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            
            if self._enable_logging:
                logger.info(
                    f"Subscribed {callback.__qualname__} to {event_type}. "
                    f"Total subscribers for {event_type}: {len(self._subscribers[event_type])}"
                )
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Name of the event type
            callback: The callback function to remove
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    if self._enable_logging:
                        logger.info(f"Unsubscribed {callback.__qualname__} from {event_type}")
                except ValueError:
                    logger.warning(f"Callback {callback.__qualname__} not found for {event_type}")
    
    def publish(self, event_type: str, event: Any) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Name of the event type (e.g., 'DataUpdateEvent')
            event: The event object to publish
        
        Example:
            event_bus.publish('DataUpdateEvent', DataUpdateEvent(...))
        """
        with self._lock:
            # Log the event
            if self._enable_logging:
                logger.info(
                    f"Publishing {event_type} to {len(self._subscribers[event_type])} subscribers"
                )
            
            # Store in history
            self._event_history.append({
                'event_type': event_type,
                'event': event,
                'timestamp': datetime.now(),
                'n_subscribers': len(self._subscribers[event_type])
            })
            
            # Notify all subscribers
            subscribers = self._subscribers[event_type].copy()
        
        # Call subscribers outside the lock to avoid deadlocks
        for callback in subscribers:
            try:
                callback(event)
                if self._enable_logging:
                    logger.debug(f"Successfully notified {callback.__qualname__}")
            except Exception as e:
                logger.error(
                    f"Error in subscriber {callback.__qualname__} for {event_type}: {e}",
                    exc_info=True
                )
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        """
        Get list of subscribers for an event type.
        
        Args:
            event_type: Name of the event type
            
        Returns:
            List of callback functions subscribed to this event type
        """
        with self._lock:
            return self._subscribers[event_type].copy()
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get event publication history.
        
        Args:
            event_type: Optional filter for specific event type
            limit: Maximum number of events to return (most recent)
            
        Returns:
            List of event history entries
        """
        with self._lock:
            history = self._event_history.copy()
        
        if event_type:
            history = [h for h in history if h['event_type'] == event_type]
        
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()
            if self._enable_logging:
                logger.info("Event history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the event bus.
        
        Returns:
            Dictionary with event bus statistics
        """
        with self._lock:
            return {
                'total_event_types': len(self._subscribers),
                'total_subscribers': sum(len(subs) for subs in self._subscribers.values()),
                'event_types': list(self._subscribers.keys()),
                'subscribers_per_type': {
                    event_type: len(subs) 
                    for event_type, subs in self._subscribers.items()
                },
                'total_events_published': len(self._event_history)
            }
    
    def __repr__(self) -> str:
        """String representation of the event bus."""
        stats = self.get_stats()
        return (
            f"EventBus(event_types={stats['total_event_types']}, "
            f"subscribers={stats['total_subscribers']}, "
            f"events_published={stats['total_events_published']})"
        )


# Global event bus instance (optional - can also create per-application)
_global_event_bus = None


def get_global_event_bus() -> EventBus:
    """
    Get or create the global event bus instance.
    
    Returns:
        Global EventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
