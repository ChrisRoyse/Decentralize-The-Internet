#!/usr/bin/env python3
import socket
import sys
import os
import requests
import neo4j

def check_neo4j():
    """Check Neo4j connection"""
    try:
        driver = neo4j.GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        )
        with driver.session() as session:
            result = session.run("RETURN 1")
            return bool(result.single())
    except:
        return False

def check_zmq_ports():
    """Check if ZMQ ports are listening"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 5556))
        sock.close()
        return result == 0
    except:
        return False

def main():
    checks = [
        check_neo4j(),
        check_zmq_ports()
    ]
    
    if all(checks):
        sys.exit(0)
    sys.exit(1)

if __name__ == "__main__":
    main() 