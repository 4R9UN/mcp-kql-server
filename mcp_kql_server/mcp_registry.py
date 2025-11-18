from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import threading

app = FastAPI()

class Server:
    """Represents a server in the MCP Registry."""
    def __init__(self, id: str, name: str, url: str, metadata: Optional[dict] = None):
        self.id = id
        self.name = name
        self.url = url
        self.metadata = metadata

class MCPRegistry:
    """In-memory MCP Registry to manage server registrations."""
    def __init__(self):
        self.servers = {}
        self.lock = threading.Lock()

    def register_server(self, server: Server):
        """Registers a new server in the registry."""
        with self.lock:
            if server.id in self.servers:
                raise HTTPException(status_code=400, detail="Server ID already registered.")
            self.servers[server.id] = server

    def unregister_server(self, server_id: str):
        """Unregisters a server from the registry by ID."""
        with self.lock:
            if server_id not in self.servers:
                raise HTTPException(status_code=404, detail="Server not found.")
            del self.servers[server_id]

    def list_servers(self) -> List[Server]:
        """Lists all registered servers."""
        with self.lock:
            return list(self.servers.values())

registry = MCPRegistry()

class ServerCreate(BaseModel):
    id: str
    name: str
    url: str
    metadata: Optional[dict] = None

@app.post("/registry/register")
def register_server(server: ServerCreate):
    new_server = Server(**server.dict())
    registry.register_server(new_server)
    return {"message": "Server registered successfully."}

@app.delete("/registry/unregister/{server_id}")
def unregister_server(server_id: str):
    registry.unregister_server(server_id)
    return {"message": "Server unregistered successfully."}

@app.get("/registry/list", response_model=List[Server])
def list_servers():
    return registry.list_servers()
