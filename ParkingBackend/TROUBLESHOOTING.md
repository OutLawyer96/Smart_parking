# 🔧 Smart Parking Backend - Troubleshooting Guide

## Common Issues & Solutions

### 1. Build Fails with Dependency Errors

**Problem**: Maven build fails with missing dependencies

**Solutions**:
```bash
# Clean Maven cache and rebuild
./mvnw clean
./mvnw install

# Or use offline mode if dependencies cached
./mvnw clean package -o

# Check Maven settings
~/.m2/settings.xml

# Verify Maven version
mvn --version  # Should be 3.6+

# Use Maven wrapper (preferred)
./mvnw clean package
```

**Prevention**: Always use `./mvnw` instead of system Maven

---

### 2. Port 8080 Already in Use

**Problem**: `Address already in use: bind`

**Solutions**:

Option A - Change the port:
```properties
# In src/main/resources/application.properties
server.port=8081
```

Option B - Kill existing process:
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>
```

Option C - Use different port on startup:
```bash
./mvnw spring-boot:run -Dspring-boot.run.arguments="--server.port=8081"
```

**Verify**: 
```bash
curl http://localhost:8080/api/parking/health
```

---

### 3. WebSocket Connection Refused

**Problem**: `WebSocket connection failed` or `Connection refused`

**Checklist**:
```bash
# 1. Verify server is running
curl http://localhost:8080/api/parking/health

# 2. Check if port is listening
lsof -i :8080
netstat -an | grep 8080

# 3. Verify WebSocket path
curl -I http://localhost:8080/ws/parking

# 4. Check firewall
# On Mac: System Preferences → Security & Privacy → Firewall
# On Linux: sudo ufw status

# 5. Check application logs
# Look for "Started ParkingBackendApplication"
# Look for "Continuous streaming started"
```

**Solution**: 
- Ensure server is running: `./mvnw spring-boot:run`
- Check logs for errors
- Verify WebSocket endpoint is registered

---

### 4. No Data Appearing on Frontend

**Problem**: Page loads but no messages or visualization

**Diagnosis**:
1. Open Browser Console (F12)
2. Check for JavaScript errors
3. Check Network tab → WebSocket status
4. Look for error messages

**Solutions**:

A. WebSocket not connecting:
```javascript
// In browser console
ws.readyState // Should be 1 (OPEN)
ws.url // Should be ws://localhost:8080/ws/parking
```

B. Fix CORS issue:
```java
// In WebSocketConfig.java
registry.addHandler(new ParkingWebSocketHandler(), "/ws/parking")
        .setAllowedOrigins("http://localhost:8080");  // Add your domain
```

C. Check if streaming is active:
```bash
curl http://localhost:8080/api/parking/status

# Response should show:
# {"connectedClients": 1, "isStreaming": true, ...}
```

D. Enable debug logging:
```properties
# In application.properties
logging.level.com.project.parkingbackend=DEBUG
```

---

### 5. High Memory Usage

**Problem**: Application using too much memory

**Investigation**:
```bash
# Monitor memory usage
top -p <PID>

# Check heap size
jps -l
jstat -gc <PID>

# Get heap dump
jmap -dump:live,format=b,file=heap.bin <PID>
```

**Solutions**:
```bash
# Limit heap size when running
./mvnw spring-boot:run -Dspring-boot.run.arguments="--server.port=8080" \
  -DargLine="-Xmx512m"

# Or in pom.xml
<maven.compiler.source>17</maven.compiler.source>
```

---

### 6. WebSocket Connections Drop Frequently

**Problem**: `WebSocket connection closed` errors

**Causes**:
- Network timeout
- Server errors
- Client browser closing tab
- Network interruption

**Solutions**:

A. Implement reconnection logic in JavaScript:
```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function reconnect() {
  if (reconnectAttempts < maxReconnectAttempts) {
    setTimeout(() => {
      console.log('Attempting to reconnect...');
      connect();
      reconnectAttempts++;
    }, 2000); // Wait 2 seconds
  }
}

ws.onclose = function() {
  reconnect();
};
```

B. Implement ping/pong in server:
```java
// In ParkingWebSocketHandler
if ("ping".equalsIgnoreCase(payload)) {
    session.sendMessage(new TextMessage("pong"));
}
```

C. Add keepalive:
```javascript
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 30000); // Every 30 seconds
```

---

### 7. JSON Parsing Errors

**Problem**: `SyntaxError: Unexpected token in JSON`

**Solution**: Check if message is valid JSON:
```javascript
ws.onmessage = function(event) {
  try {
    const data = JSON.parse(event.data);
    console.log('Valid JSON:', data);
  } catch (e) {
    console.error('Invalid JSON:', event.data);
    console.error('Error:', e);
  }
};
```

**Debug**:
```bash
# In browser console, intercept and log raw data
const originalOnMessage = ws.onmessage;
ws.onmessage = function(event) {
  console.log('Raw message:', event.data);
  originalOnMessage.call(this, event);
};
```

---

### 8. Canvas Not Rendering

**Problem**: Canvas appears blank or black

**Diagnosis**:
1. Open browser console
2. Check canvas size: `canvas.width`, `canvas.height`
3. Check if data is being received

**Solutions**:
```javascript
// Verify canvas element exists
const canvas = document.getElementById('parkingCanvas');
console.log('Canvas:', canvas);
console.log('Width:', canvas.width);
console.log('Height:', canvas.height);

// Verify context
const ctx = canvas.getContext('2d');
console.log('Context:', ctx);

// Test drawing
ctx.fillStyle = '#ff0000';
ctx.fillRect(10, 10, 50, 50);
```

---

### 9. Messages Not Broadcasting to All Clients

**Problem**: One client connected, other clients don't receive messages

**Diagnosis**:
```bash
# Check how many clients are connected
curl http://localhost:8080/api/parking/status
```

**Solutions**:

A. Verify client connections:
```java
// Add logging to ParkingWebSocketHandler
@Override
public void afterConnectionEstablished(WebSocketSession session) {
    sessions.add(session);
    logger.info("Client connected. Total: {}", sessions.size()); // Check log
}
```

B. Check if streaming is running:
```bash
curl http://localhost:8080/api/parking/status
# Check: "isStreaming": true
```

C. Enable debug logging:
```properties
logging.level.com.project.parkingbackend.websocket=DEBUG
```

---

### 10. Application Crashes on Startup

**Problem**: `Exception`, `Error`, or immediate exit

**Solutions**:

1. Check logs for error message:
```bash
./mvnw spring-boot:run 2>&1 | grep -i error
```

2. Common causes:
   - Port already in use → Change port
   - Java version mismatch → Update Java
   - Missing dependencies → Run `./mvnw install`
   - Compilation errors → Run `./mvnw clean compile`

3. Full error trace:
```bash
./mvnw spring-boot:run -X  # Enable debug
```

---

### 11. Slow Performance / High Latency

**Problem**: Messages arrive slowly or with delay

**Investigation**:
```javascript
// Measure message latency
let lastTime = Date.now();
ws.onmessage = function(event) {
  const latency = Date.now() - lastTime;
  console.log('Latency:', latency, 'ms');
  lastTime = Date.now();
};
```

**Solutions**:

A. Reduce streaming frequency:
```java
// In ParkingWebSocketHandler
Thread.sleep(200);  // From 100ms to 200ms
```

B. Check network:
```bash
# On Mac
networkQuality

# Check bandwidth
# Test latency with ping
ping localhost
```

C. Monitor CPU:
```bash
# Check CPU usage
top
# Java should use <20% CPU
```

D. Increase heap size:
```bash
java -Xmx1g -Xms512m -jar target/ParkingBackend-0.0.1-SNAPSHOT.jar
```

---

### 12. CORS Issues (When Deployed)

**Problem**: `No 'Access-Control-Allow-Origin' header`

**Solution**:
```java
// In WebSocketConfig.java
registry.addHandler(new ParkingWebSocketHandler(), "/ws/parking")
        .setAllowedOrigins(
            "https://yourdomain.com",
            "https://app.yourdomain.com"
        );
```

For development (allow all):
```java
.setAllowedOrigins("*");  // Only for development!
```

---

### 13. Data Validation Errors

**Problem**: Invalid data in response

**Solutions**:

A. Validate on client side:
```javascript
function validateData(data) {
  if (!Array.isArray(data.path)) throw new Error('Invalid path');
  if (!Array.isArray(data.obstacles)) throw new Error('Invalid obstacles');
  if (!data.targetSlot) throw new Error('Invalid targetSlot');
  return true;
}

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (validateData(data)) {
    updateUI(data);
  }
};
```

B. Check server data generation:
```bash
# Enable debug in ParkingDataService
logging.level.com.project.parkingbackend.service=DEBUG
```

---

### 14. Docker-Related Issues

**Problem**: Docker container fails or can't connect

**Solutions**:

A. Check Docker logs:
```bash
docker logs <container_id>
```

B. Verify port mapping:
```bash
docker run -p 8080:8080 parking-backend
# Should map port 8080
```

C. Check network:
```bash
docker network ls
docker inspect <network_id>
```

---

### 15. Security Warnings

**Problem**: Security/SSL warnings

**Solutions for Development**:
- Ignore warnings (development only)
- Use self-signed certificate for testing

**Solutions for Production**:
```bash
# Get SSL certificate
certbot certonly --standalone -d yourdomain.com

# Add to Spring:
server.ssl.key-store=classpath:keystore.p12
server.ssl.key-store-password=<password>
server.ssl.keyStoreType=PKCS12
```

---

## Performance Optimization Tips

### 1. Reduce Message Frequency
```java
Thread.sleep(200);  // Send every 200ms instead of 100ms
```

### 2. Limit Data Size
```java
// Send only essential coordinates
List<Coordinate> importantPoints = path.stream()
    .filter(c -> /* keep only key points */)
    .collect(Collectors.toList());
```

### 3. Enable Compression
```properties
server.compression.enabled=true
server.compression.min-response-size=1024
```

### 4. Optimize Canvas Rendering
```javascript
// Only update when data changes
let lastData = null;
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (JSON.stringify(data) !== JSON.stringify(lastData)) {
    drawParking(data);
    lastData = data;
  }
};
```

---

## Debug Logging

### Enable Debug Logs
```properties
# application.properties
logging.level.root=WARN
logging.level.com.project.parkingbackend=DEBUG
logging.level.org.springframework.web=DEBUG
```

### Redirect Logs to File
```properties
logging.file.name=logs/parking-backend.log
logging.file.max-size=10MB
logging.file.max-history=10
```

---

## Health Checks

### Basic Checks
```bash
# Server is running
curl http://localhost:8080/api/parking/health

# Streaming is active
curl http://localhost:8080/api/parking/status

# WebSocket is accessible
wscat -c ws://localhost:8080/ws/parking
```

### Advanced Checks
```bash
# View JVM stats
jstat -gc <PID>

# View threads
jstack <PID>

# View heap
jmap -heap <PID>
```

---

## Getting Help

If you encounter an issue not listed here:

1. **Check Logs**
   - Terminal where server is running
   - Browser console (F12)
   - Application logs

2. **Review Documentation**
   - IMPLEMENTATION.md - Technical details
   - TESTING_GUIDE.md - Testing procedures
   - ARCHITECTURE.md - System design

3. **Enable Debug Logging**
   ```properties
   logging.level.com.project.parkingbackend=TRACE
   ```

4. **Search for Similar Issues**
   - GitHub issues
   - Stack Overflow
   - Spring Boot documentation

---

**Still stuck?** Refer to the implementation source code. Every class has inline comments explaining the logic.

**Last Updated**: 2026-03-10

