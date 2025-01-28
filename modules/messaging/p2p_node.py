import time
import random
from typing import Set, Dict, Any
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    node_id: str
    address: str
    last_seen: float
    
class P2PNode:
    """
    Manages peer discovery and maintains connections in the P2P network
    """
    
    def __init__(self, config, message_bus):
        self.config = config
        self.message_bus = message_bus
        self.node_id = config["node"]["id"]
        
        # Known peers
        self.peers: Dict[str, PeerInfo] = {}
        
        # Subscribe to peer discovery messages
        self.message_bus.subscribe("peer_discovery", self._handle_peer_discovery)
        self.message_bus.subscribe("peer_announce", self._handle_peer_announce)
        
        # Start periodic peer discovery
        self._start_discovery()
        
    def _start_discovery(self):
        """Start periodic peer discovery process"""
        import threading
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        
    def _discovery_loop(self):
        """Periodically announce presence and clean up stale peers"""
        while True:
            try:
                # Announce ourselves
                self._announce_presence()
                
                # Clean up stale peers
                self._cleanup_peers()
                
                # Sleep for random interval (5-10 seconds)
                time.sleep(5 + random.random() * 5)
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(5)  # Back off on error
                
    def _announce_presence(self):
        """Announce our presence to the network"""
        announcement = {
            "node_id": self.node_id,
            "address": f"127.0.0.1:{self.message_bus.pub_port}",  # In real usage, get actual IP
            "timestamp": time.time()
        }
        self.message_bus.publish("peer_announce", announcement)
        
    def _handle_peer_announce(self, message: Dict[str, Any]):
        """Handle peer announcement message"""
        try:
            node_id = message["node_id"]
            address = message["address"]
            timestamp = message["timestamp"]
            
            # Skip if it's our own announcement
            if node_id == self.node_id:
                return
                
            # Update peer info
            self.peers[node_id] = PeerInfo(
                node_id=node_id,
                address=address,
                last_seen=timestamp
            )
            
            # Connect to new peer if we haven't already
            self.message_bus.connect_to_peer(address)
            
        except Exception as e:
            logger.error(f"Error handling peer announcement: {e}")
            
    def _handle_peer_discovery(self, message: Dict[str, Any]):
        """Handle peer discovery request"""
        try:
            # Respond with our peer list
            response = {
                "node_id": self.node_id,
                "peers": [
                    {
                        "node_id": p.node_id,
                        "address": p.address
                    }
                    for p in self.peers.values()
                ]
            }
            self.message_bus.publish("peer_discovery_response", response)
            
        except Exception as e:
            logger.error(f"Error handling peer discovery: {e}")
            
    def _cleanup_peers(self):
        """Remove peers that haven't been seen recently"""
        current_time = time.time()
        stale_timeout = 60  # 60 seconds
        
        stale_peers = [
            node_id for node_id, peer in self.peers.items()
            if current_time - peer.last_seen > stale_timeout
        ]
        
        for node_id in stale_peers:
            logger.info(f"Removing stale peer {node_id}")
            del self.peers[node_id]
            
    def get_peer_count(self) -> int:
        """Get number of active peers"""
        return len(self.peers)
        
    def get_peer_addresses(self) -> Set[str]:
        """Get set of peer addresses"""
        return {peer.address for peer in self.peers.values()} 