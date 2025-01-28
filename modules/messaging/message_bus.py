import zmq
import json
import threading
import logging
from typing import Callable, Dict, Any
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Message:
    topic: str
    payload: Dict[str, Any]
    message_id: str = None
    sender_id: str = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

class MessageBus:
    """
    Handles peer-to-peer communication using ZeroMQ PUB/SUB pattern
    """
    
    def __init__(self, config):
        self.config = config
        self.node_id = config["node"]["id"]
        
        # Initialize ZMQ context
        self.context = zmq.Context()
        
        # Publisher socket
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_port = self._bind_to_random_port(self.pub_socket)
        
        # Subscriber socket
        self.sub_socket = self.context.socket(zmq.SUB)
        
        # Callbacks for message topics
        self.callbacks: Dict[str, Callable] = {}
        
        # Track seen message IDs to prevent duplicates
        self.seen_messages = set()
        
        # Start listener thread
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        
        logger.info(f"MessageBus initialized for node {self.node_id} on port {self.pub_port}")

    def _bind_to_random_port(self, socket, min_port=49152, max_port=65535):
        """Bind socket to random available port"""
        while True:
            try:
                port = socket.bind_to_random_port(
                    "tcp://*",
                    min_port=min_port,
                    max_port=max_port
                )
                return port
            except zmq.ZMQError as e:
                logger.warning(f"Failed to bind to port: {e}")
                continue

    def connect_to_peer(self, peer_address: str):
        """Connect subscriber socket to a peer's publisher"""
        try:
            self.sub_socket.connect(f"tcp://{peer_address}")
            logger.info(f"Connected to peer at {peer_address}")
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_address}: {e}")

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic with callback"""
        try:
            # Subscribe to topic
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            
            # Store callback
            self.callbacks[topic] = callback
            logger.debug(f"Subscribed to topic: {topic}")
            
        except Exception as e:
            logger.error(f"Error subscribing to {topic}: {e}")

    def publish(self, topic: str, data: Dict[str, Any]):
        """Publish message to topic"""
        try:
            message = Message(
                topic=topic,
                payload=data,
                sender_id=self.node_id
            )
            
            # Serialize message
            msg_str = f"{topic} {json.dumps(message.__dict__)}"
            
            # Send
            self.pub_socket.send_string(msg_str)
            logger.debug(f"Published message to topic {topic}")
            
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")

    def _listen_loop(self):
        """Background thread to listen for messages"""
        while self.running:
            try:
                # Receive message with timeout
                if self.sub_socket.poll(timeout=1000):
                    msg_str = self.sub_socket.recv_string()
                    self._handle_message(msg_str)
            except Exception as e:
                logger.error(f"Error in listener loop: {e}")
                continue

    def _handle_message(self, msg_str: str):
        """Process received message"""
        try:
            # Parse topic and payload
            topic, json_str = msg_str.split(" ", 1)
            message = json.loads(json_str)
            
            # Skip if we've seen this message
            if message["message_id"] in self.seen_messages:
                return
                
            # Skip messages from self
            if message["sender_id"] == self.node_id:
                return
                
            # Mark message as seen
            self.seen_messages.add(message["message_id"])
            
            # Call topic callback if exists
            if topic in self.callbacks:
                self.callbacks[topic](message["payload"])
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def close(self):
        """Clean shutdown"""
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term() 