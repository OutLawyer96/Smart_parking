# 🚀 Quick Start Guide - Smart Parking Backend

## ⚡ 5-Minute Setup

### 1. Start the Server
```bash
cd /Users/hitendrasingh/Desktop/ParkingBackend
./mvnw spring-boot:run
```

You should see:
```
Started ParkingBackendApplication in X seconds
```

### 2. Open the Frontend
Navigate to: http://localhost:8080

### 3. Connect to WebSocket
Click "Connect to Server" button on the web interface

### 4. Watch Real-time Data
- **Canvas**: Shows the parking area with path, obstacles, and target slot
- **Live Data Stream**: Displays the JSON response in real-time
- **Console**: Shows connection status and activity

## 🎯 What You're Seeing

The system sends parking navigation data that includes:

1. **Path** - Coordinates the vehicle should follow
2. **Obstacles** - Pillars and other parked vehicles to avoid
3. **Target Slot** - The destination parking spot

Data updates **10 times per second** (every 100ms).

## 📊 Visualization Elements

| Element | Color | Meaning |
|---------|-------|---------|
| Dark Gray Boxes | Static | Fixed pillars/structures |
| Red Boxes | Dynamic | Other parked vehicles |
| Green Box | Highlight | Target parking slot |
| Blue Path | Line | Recommended vehicle path |
| Purple Dots | Points | Path waypoints |

## 🔍 Check Server Status

Visit: http://localhost:8080/api/parking/status

Response shows:
- Connected clients
- Streaming status
- WebSocket URL

## 💻 For Frontend Developers

Include this WebSocket connection in your app:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/parking');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  // Draw path: data.path
  // Draw obstacles: data.obstacles
  // Draw target: data.targetSlot
};
```

## 🛠️ For ML Engineers

To integrate your ML model:

1. Open `src/main/java/com/project/parkingbackend/service/ParkingDataService.java`
2. Modify the `generateParkingData()` method
3. Replace mock data generation with your ML model calls
4. Rebuild: `./mvnw clean package`

## ⚙️ Adjust Update Frequency

In `ParkingWebSocketHandler.java`, change this line:
```java
Thread.sleep(100);  // 100ms = 10 updates/second
```

To:
```java
Thread.sleep(50);   // 50ms = 20 updates/second
```

## 🧪 Test with Different Clients

Open multiple browser tabs to `http://localhost:8080` - the server will broadcast to all connected clients simultaneously!

## 📱 Use Custom Client

Connect any WebSocket client to: `ws://localhost:8080/ws/parking`

Example with curl-websocket tools or custom apps.

## 🐛 Debug Issues

### Check Server Logs
```bash
# Logs appear in the terminal where you ran spring-boot:run
# Look for DEBUG messages from com.project.parkingbackend
```

### Browser Console
Press F12 in the browser and check the Console tab for any JavaScript errors.

### Monitor Connections
Visit: http://localhost:8080/api/parking/status

Should show:
- `connectedClients`: > 0
- `isStreaming`: true

## 🎓 Next Steps

1. **Customize Data**: Modify `ParkingDataService` to generate realistic parking data
2. **Add Frontend**: Build your own UI consuming the WebSocket
3. **Integrate ML**: Replace mock data with actual ML model predictions
4. **Deploy**: Run on production server with proper SSL/TLS

## 📚 Full Documentation

See `IMPLEMENTATION.md` for complete API documentation and configuration options.

---

**That's it! Your smart parking system is running! 🎉**

