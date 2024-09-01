import asyncio
import websockets
from aioconsole import ainput

async def communicate():
    uri = "ws://127.0.0.1:8765"
    async with websockets.connect(uri) as websocket:
        print("Connected to the server")
        try:
            # Task for receiving messages from the server
            async def receive_messages():
                while True:
                    try:
                        message = await websocket.recv()
                        print(f"Received message from server: {message}")
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed by the server")
                        break
                    except Exception as e:
                        print(f"Error receiving message: {e}")

            # Task for sending messages to the server
            async def send_messages():
                while True:
                    try:
                        message = await ainput("Enter message to send to server: ")
                        await websocket.send(message)
                        print(f"Sent message to server: {message}")
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed by the server")
                        break
                    except Exception as e:
                        print(f"Error sending message: {e}")

            # Run both tasks concurrently
            await asyncio.gather(receive_messages(), send_messages())
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(communicate())