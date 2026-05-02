import asyncio

class WebSocketLogger:
    def __init__(self):
        self.connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    def log(self, message: str):
        print(message) # También lo imprime en la terminal real de Ubuntu
        # Como los modelos bloquean el hilo principal, usamos el loop asíncrono para enviar
        try:
            loop = asyncio.get_running_loop()
            for connection in self.connections:
                loop.create_task(connection.send_text(message))
        except RuntimeError:
            pass # Si no hay loop corriendo en este hilo

ws_logger = WebSocketLogger()