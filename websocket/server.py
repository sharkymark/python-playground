import asyncio
import websockets
import sys

async def echo(websocket, path):
    print("Client connected")
    try:
        # Task for receiving messages from the client
        async def receive_messages():
            async for message in websocket:
                print(f"Received message from client: {message}")
                response = f"Echo: {message}"
                await websocket.send(response)
                print(f"Sent response to client: {response}")

        # Task for sending messages to the client
        async def send_messages():
            while True:
                # Example message to send to the client
                message = "Hello from server"
                await websocket.send(message)
                print(f"Sent message to client: {message}")
                await asyncio.sleep(5)  # Wait for 5 seconds before sending the next message

        # Run both tasks concurrently
        await asyncio.gather(receive_messages(), send_messages())
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

async def main():
    async with websockets.serve(echo, "127.0.0.1", 8765):
        print("WebSocket server started at ws://127.0.0.1:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())