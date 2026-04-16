# 🏗️ Smart Parking Backend - Architecture & Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────┐         ┌─────────────────────────┐        │
│  │   Web Browser (index.html)  │         │   Mobile App / Custom   │        │
│  │                             │         │   Client Application    │        │
│  │  - Canvas Visualization     │         │                         │        │
│  │  - Connection Management    │         │  - WebSocket Client     │        │
│  │  - Real-time Data Display   │         │  - Data Processing      │        │
│  └────────────┬────────────────┘         └────────────┬────────────┘        │
│               │                                       │                      │
└───────────────┼───────────────────────────────────────┼──────────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │ WebSocket Protocol
                                    │ ws://localhost:8080/ws/parking
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRANSPORT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Spring WebSocket (TextWebSocketHandler)                                     │
│  - Connection Lifecycle Management                                           │
│  - Message Broadcasting                                                      │
│  - Error Handling & Recovery                                                 │
│                                                                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER (Server)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  ParkingWebSocketHandler                            │   │
│  │                                                                     │   │
│  │  - afterConnectionEstablished()                                    │   │
│  │  - handleTextMessage()                                             │   │
│  │  - afterConnectionClosed()                                         │   │
│  │  - broadcastParkingData()                                          │   │
│  │  - startContinuousStreaming()                                      │   │
│  │  - stopContinuousStreaming()                                       │   │
│  │                                                                     │   │
│  │  Session Management:                                               │   │
│  │  - CopyOnWriteArrayList<WebSocketSession> sessions                │   │
│  │  - Thread-safe concurrent operations                               │   │
│  │  - Automatic cleanup on disconnect                                 │   │
│  └──────────────┬──────────────────────────┬──────────────────────────┘   │
│                 │                          │                              │
│   ┌─────────────▼──────────┐   ┌──────────▼──────────────┐              │
│   │  WebSocketConfig       │   │  ParkingController      │              │
│   │                        │   │  (REST API)             │              │
│   │  - Bean Registration   │   │                        │              │
│   │  - Handler Mapping     │   │  GET /api/parking/     │              │
│   │  - CORS Configuration  │   │   - health             │              │
│   │                        │   │   - status             │              │
│   └────────────────────────┘   └────────────────────────┘              │
│                                                                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUSINESS LOGIC LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    ParkingDataService                                │  │
│  │                                                                      │  │
│  │  - generateParkingData()                                            │  │
│  │    └─> Generates:                                                   │  │
│  │        • Path coordinates (List<Coordinate>)                        │  │
│  │        • Obstacles (List<Obstacle>)                                 │  │
│  │        • Target slot (TargetSlot)                                   │  │
│  │                                                                      │  │
│  │  - generatePath()                                                   │  │
│  │  - getStaticObstacles()                                             │  │
│  │  - getTargetSlot()                                                  │  │
│  │  - hasReachedTarget()                                               │  │
│  │  - addRealisticVariations()                                         │  │
│  │                                                                      │  │
│  │  TODO: Integrate with ML Model Service for real predictions         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MLParkingDataService (Example)                    │  │
│  │                                                                      │  │
│  │  - generateParkingDataFromML()                                      │  │
│  │  - convertMLOutputToParkingResponse()                               │  │
│  │                                                                      │  │
│  │  Example structures:                                                │  │
│  │  - MLModelInput  (vehicle position, heading, obstacles, etc.)       │  │
│  │  - MLModelOutput (predicted path, confidence, steering commands)    │  │
│  │  - SteeringCommand (timestamp, angle, acceleration)                 │  │
│  │                                                                      │  │
│  │  Can integrate with:                                                │  │
│  │  - REST API calls to Python ML service                              │  │
│  │  - gRPC calls to distributed ML systems                             │  │
│  │  - In-process ML libraries                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA MODEL LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │   Coordinate         │  │   Rectangle      │  │   TargetSlot          │ │
│  │                      │  │                  │  │                       │ │
│  │  - x: double         │  │  - x: double     │  │  - x: double          │ │
│  │  - y: double         │  │  - y: double     │  │  - y: double          │ │
│  │                      │  │  - width: double │  │  - width: double      │ │
│  │  Getters/Setters     │  │  - height: double│  │  - length: double     │ │
│  │  toString()          │  │                  │  │                       │ │
│  │                      │  │  Getters/Setters │  │  Getters/Setters      │ │
│  │                      │  │  toString()      │  │  toString()           │ │
│  └──────────────────────┘  └──────────────────┘  └───────────────────────┘ │
│                                                                               │
│  ┌──────────────────────┐                   ┌──────────────────────────┐   │
│  │   Obstacle           │                   │  ParkingResponse         │   │
│  │                      │                   │                          │   │
│  │  - id: String        │                   │  - path: List<Coord>     │   │
│  │  - rect: Rectangle   │                   │  - obstacles: List<Obs>  │   │
│  │  - isDynamic: bool   │                   │  - targetSlot: Target    │   │
│  │                      │                   │                          │   │
│  │  Getters/Setters     │                   │  Getters/Setters         │   │
│  │  toString()          │                   │  toString()              │   │
│  └──────────────────────┘                   └──────────────────────────┘   │
│                                                                               │
│  (All with Jackson @JsonProperty annotations for serialization)              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Message Flow Diagram

```
TIME AXIS →

Client                                    Server
  │                                        │
  │  1. WebSocket Connect                  │
  ├───────────────────────────────────────>│
  │                                        │ afterConnectionEstablished()
  │                                        │ - Add to sessions list
  │                                        │ - Start streaming thread (first client)
  │  2. Initial Data                       │
  │<───────────────────────────────────────┤
  │ {path, obstacles, targetSlot}          │
  │                                        │
  │  3. Streaming Loop (every 100ms)       │
  │<───────────────────────────────────────┤
  │ {path, obstacles, targetSlot}          │
  │                                        │
  │<───────────────────────────────────────┤
  │ {path, obstacles, targetSlot}          │
  │                                        │
  │<───────────────────────────────────────┤
  │ {path, obstacles, targetSlot}          │
  │                                        │
  │  4. Optional: Send Command              │
  ├───────────────────────────────────────>│
  │ "start" / "stop" / "ping"              │ handleTextMessage()
  │                                        │
  │<───────────────────────────────────────┤
  │ "pong" (if ping)                       │
  │                                        │
  │  5. WebSocket Disconnect               │
  ├───────────────────────────────────────>│
  │                                        │ afterConnectionClosed()
  │                                        │ - Remove from sessions
  │                                        │ - Stop streaming (if last client)
  │                                        │
```

## Data Generation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                   Streaming Thread (1 per server)                │
│                                                                  │
│   while (isStreaming && !sessions.isEmpty()) {                  │
│      ┌─────────────────────────────────────────────────────┐   │
│      │  ParkingDataService.generateParkingData()          │   │
│      │                                                     │   │
│      │  1. generatePath()                                 │   │
│      │     - Static base path coordinates                 │   │
│      │     - Add realistic variations                     │   │
│      │     - Return List<Coordinate>                      │   │
│      │                                                     │   │
│      │  2. getStaticObstacles()                           │   │
│      │     - pillar-1, pillar-2 (isDynamic=false)        │   │
│      │     - parked-1, parked-2 (isDynamic=true)         │   │
│      │     - Return List<Obstacle>                        │   │
│      │                                                     │   │
│      │  3. getTargetSlot()                                │   │
│      │     - Target parking position                      │   │
│      │     - Return TargetSlot                            │   │
│      │                                                     │   │
│      │  Returns: ParkingResponse                          │   │
│      └─────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│      ┌─────────────────────────────────────────────────────┐   │
│      │  broadcastParkingData(parkingData)                 │   │
│      │                                                     │   │
│      │  1. ObjectMapper.writeValueAsString()              │   │
│      │     - Serialize to JSON                            │   │
│      │                                                     │   │
│      │  2. For each WebSocketSession in sessions:         │   │
│      │     - If session.isOpen():                         │   │
│      │       └─ session.sendMessage(jsonMessage)          │   │
│      │     - If error: remove from sessions               │   │
│      │                                                     │   │
│      └─────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│      Thread.sleep(100);  // 10 messages/second                 │
│   }                                                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Thread Management

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Spring Thread Pool                       │
│  (Handles WebSocket connection/lifecycle events)                │
└─────────────────────────────────────────────────────────────────┘
           │
           ├──> afterConnectionEstablished()  [Thread 1]
           │
           ├──> handleTextMessage()          [Thread 2]
           │
           ├──> afterConnectionClosed()      [Thread 3]
           │
           └──> Creates: streamingThread
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Dedicated Streaming Thread (1)                     │
│      (ParkingDataStreaming - continuous generation/broadcast)   │
│                                                                  │
│  Purpose: Generate and broadcast data to all clients            │
│  Start: When first client connects                              │
│  Stop: When last client disconnects                             │
│  Interval: 100ms between broadcasts                             │
│                                                                  │
│  Thread-safe access to:                                         │
│  - CopyOnWriteArrayList<WebSocketSession> sessions             │
│  - volatile boolean isStreaming                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Connection Lifecycle

```
CLIENT PERSPECTIVE              SERVER PERSPECTIVE
─────────────────────           ──────────────────

Connect()                  ┌─-> WebSocket Accept
  │                        │
  │                   Session Created
  │                        │
  ├─────────────────────>──┤
                     │ afterConnectionEstablished()
                     │ • Add to sessions list
                     │ • Check if first client
                     │ • Start streaming (if needed)
  │                  │
  │<─────────────────┤
  │  Initial Data        │ Send current data
  │                      │
  ├─────────────────────────────────────────────┐
  │                                             │ Streaming Loop
  │<────────────────────────────────────────────┤ (every 100ms)
  │  Update Data                                │
  │<────────────────────────────────────────────┤
  │  Update Data                                │
  │<────────────────────────────────────────────┤
  │  Update Data                                │
  │                                             │
  │ (Optional: Send Command)                    │
  ├─────────────────────────────────────────────┤
  │ "start"/"stop"/"ping"                      │ handleTextMessage()
  │                                             │
  Disconnect()               │ afterConnectionClosed()
     │                       │ • Remove from sessions
     └───────────────────────┤ • Check if last client
                            │ • Stop streaming (if needed)
                            │
                       Session Closed
```

## Class Relationships

```
                    ┌──────────────────┐
                    │  Spring Framework │
                    │  (WebSocket)      │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  WebSocketConfig  │
                    │  @Configuration   │
                    │  @EnableWebSocket │
                    └────────┬──────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ParkingWebSocketHandler   │    │  RestController          │
│extends                   │    │  ParkingController       │
│  TextWebSocketHandler    │    │                          │
│                          │    │  @GetMapping("/status")  │
│ Static methods:          │    │  @GetMapping("/health")  │
│ - getConnectedClients()  │    │                          │
│ - isStreamingActive()    │    └──────────────────────────┘
│                          │
│ Uses:                    │
│ - ParkingDataService     │
│ - ObjectMapper (Jackson) │
│ - CopyOnWriteArrayList   │
└──────────────────────────┘

         ┌──────────────────────┐
         │ ParkingDataService   │
         │ @Service             │
         │                      │
         │ - generateParkingData()
         │ - generatePath()     │
         │ - getObstacles()     │
         │ - getTargetSlot()    │
         │ - hasReachedTarget() │
         └──────────────────────┘

         ┌──────────────────────┐
         │ MLParkingDataService │
         │ (Example)            │
         │                      │
         │ - MLModelInput       │
         │ - MLModelOutput      │
         │ - SteeringCommand    │
         └──────────────────────┘
         
         ┌──────────────────────┐
         │ Model Classes        │
         │ (@JsonProperty)      │
         │                      │
         │ - Coordinate         │
         │ - Rectangle          │
         │ - Obstacle           │
         │ - TargetSlot         │
         │ - ParkingResponse    │
         └──────────────────────┘
```

## Concurrency & Thread Safety

```
┌─────────────────────────────────────────────────────────┐
│                 Thread Safety Mechanisms                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. CopyOnWriteArrayList<WebSocketSession>             │
│     - Thread-safe iteration                            │
│     - Safe during concurrent add/remove                │
│     - Each write creates a copy                        │
│                                                          │
│  2. Synchronized Methods                               │
│     - startContinuousStreaming()                       │
│     - stopContinuousStreaming()                        │
│     - Prevents duplicate streaming threads             │
│                                                          │
│  3. Volatile Variables                                  │
│     - volatile boolean isStreaming                     │
│     - Ensures visibility across threads                │
│                                                          │
│  4. Thread.interrupt()                                  │
│     - Graceful shutdown of streaming thread            │
│                                                          │
│  Safe Operations:                                       │
│  ✓ Multiple clients reading simultaneously             │
│  ✓ Adding/removing clients during streaming            │
│  ✓ Starting/stopping streaming                         │
│  ✓ Exception handling in message sending               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Error Handling Strategy

```
┌──────────────────────────────────────────────────────────┐
│                   Error Handling Flow                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. Connection Errors                                    │
│     └─> Caught in afterConnectionEstablished()          │
│     └─> Logged with session ID                          │
│     └─> Connection rejected (naturally)                 │
│                                                           │
│  2. Message Sending Errors                              │
│     └─> Caught in broadcastParkingData()               │
│     └─> Failed session removed from list                │
│     └─> Error logged with session ID                    │
│     └─> Continue with other sessions                    │
│                                                           │
│  3. JSON Serialization Errors                           │
│     └─> Caught in broadcastParkingData()               │
│     └─> Error logged (not per-session)                  │
│     └─> Fallback to mock data                           │
│                                                           │
│  4. Threading Errors                                     │
│     └─> InterruptedException caught                     │
│     └─> Thread interrupted                              │
│     └─> isStreaming set to false                        │
│     └─> Streaming stops gracefully                      │
│                                                           │
│  5. Unexpected Errors                                    │
│     └─> Try-catch in each method                        │
│     └─> Logged with stack trace                         │
│     └─> Service continues running                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

This architecture is designed for:
- **Scalability**: Multiple clients, efficient broadcasting
- **Reliability**: Error handling and recovery
- **Performance**: Minimal latency, configurable update rate
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to integrate ML models and sensors

**Architecture Version**: 1.0
**Last Updated**: 2026-03-10

